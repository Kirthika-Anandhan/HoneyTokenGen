"""
Threat attribution & attack profiling inference.

Builds temporal snapshots from events, optionally runs HGT+Transformer model,
and always augments with rule-based heuristics for interpretability.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_AG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "attack_graph_module"))
if _AG not in sys.path:
    sys.path.insert(0, _AG)

from data.event_processor import (  # noqa: E402
    SyntheticAttackGenerator,
    TemporalGraphBuilder,
)

_TA = os.path.dirname(os.path.abspath(__file__))

from threat_attribution_module.models.hgt_sequence_model import (  # noqa: E402
    ThreatAttributionSequenceModel,
    ATTRIBUTION_LABELS,
    PROFILE_LABELS,
)


def _events_in_snapshot(events: List[Dict], t_start: float, t_end: float) -> List[Dict]:
    return [e for e in events if t_start <= e["timestamp"] < t_end]


def _profile_from_events(evs: List[Dict]) -> Tuple[str, float]:
    """Rule-based dominant stage for one window."""
    if not evs:
        return "benign", 0.55
    actions = [e.get("action", "") for e in evs]
    score = {}
    if any(a == "scan" for a in actions):
        score["reconnaissance"] = 0.9
    if any(a == "login" for a in actions) or any(a == "brute_force" for a in actions):
        score["reconnaissance"] = max(score.get("reconnaissance", 0), 0.65)
    if any(a == "lateral_move" for a in actions):
        score["lateral_movement"] = 0.88
    if any(a == "access_honeytoken" for a in actions):
        score["lateral_movement"] = max(score.get("lateral_movement", 0), 0.82)
    if any(a == "escalate" for a in actions):
        score["privilege_escalation"] = 0.9
    if any(a == "exfiltrate" for a in actions):
        score["exfiltration"] = 0.92
    benign_like = sum(1 for a in actions if a in ("connect", "read", "login") and "scan" not in actions)
    if benign_like >= len(actions) * 0.7 and not score:
        return "benign", 0.75
    if not score:
        return "benign", 0.6
    best = max(score, key=score.get)
    return best, score[best]


def _campaign_attribution_heuristic(events: List[Dict]) -> Tuple[str, float, Dict[str, float]]:
    labels = [ATTRIBUTION_LABELS[i] for i in range(5)]
    if not events:
        return ATTRIBUTION_LABELS[0], 0.5, {k: 0.2 for k in labels}
    scans = sum(1 for e in events if e.get("action") == "scan")
    lateral = sum(1 for e in events if e.get("action") == "lateral_move")
    honey = sum(1 for e in events if e.get("action") == "access_honeytoken")
    exfil = sum(1 for e in events if e.get("action") == "exfiltrate")
    esc = sum(1 for e in events if e.get("action") == "escalate")
    procs = [e.get("process", "") for e in events]
    auto = sum(1 for p in procs if p in ("nmap", "hydra", "masscan", "metasploit"))

    dist = {k: 0.05 for k in labels}
    if scans > 5 or auto > 3:
        dist[ATTRIBUTION_LABELS[1]] += 0.45
        dist[ATTRIBUTION_LABELS[4]] += 0.25
    if lateral > 2 or honey > 0:
        dist[ATTRIBUTION_LABELS[2]] += 0.4
    if esc > 1:
        dist[ATTRIBUTION_LABELS[3]] += 0.35
    if exfil > 0:
        dist[ATTRIBUTION_LABELS[2]] += 0.2
        dist[ATTRIBUTION_LABELS[1]] += 0.15

    s = sum(dist.values())
    dist = {k: round(v / s, 4) for k, v in dist.items()}
    primary = max(dist, key=dist.get)
    conf = dist[primary]
    return primary, conf, dist


def _softmax_dict(logits: torch.Tensor, labels: Dict[int, str]) -> Dict[str, float]:
    p = F.softmax(logits.squeeze(0), dim=-1)
    return {labels[i]: round(p[i].item(), 4) for i in range(len(labels))}


def run_attribution_pipeline(
    events: List[Dict],
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    window_size_sec: int = 300,
    slide_step_sec: int = 60,
) -> Dict[str, Any]:
    builder = TemporalGraphBuilder(window_size_sec, slide_step_sec)
    snapshots = builder.build_snapshots(events)
    if not snapshots:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "status": "empty",
            "message": "No graph snapshots produced from events.",
            "event_count": len(events),
        }

    per_snap_events = [
        _events_in_snapshot(events, float(snap.t_start), float(snap.t_end)) for snap in snapshots
    ]

    heur_profile = []
    for evs in per_snap_events:
        stage, conf = _profile_from_events(evs)
        heur_profile.append({"stage": stage, "confidence": round(conf, 3), "source": "heuristic"})

    att_label, att_conf, att_dist = _campaign_attribution_heuristic(events)

    next_heur = []
    for i in range(len(heur_profile)):
        nxt = heur_profile[i + 1]["stage"] if i + 1 < len(heur_profile) else heur_profile[i]["stage"]
        next_heur.append(
            {"predicted_next_stage": nxt, "confidence": 0.55, "source": "heuristic_rollforward"}
        )

    neural_block: Dict[str, Any] = {"available": False}
    ckpt = checkpoint or os.path.join(_TA, "checkpoints", "best_hgt_seq.pt")

    if os.path.isfile(ckpt):
        try:
            dev = torch.device(device)
            model = ThreatAttributionSequenceModel()
            state = torch.load(ckpt, map_location=dev)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                a_log, p_log, n_log = model(snapshots, device=dev)
            neural_block = {
                "available": True,
                "checkpoint": ckpt,
                "attribution_distribution": _softmax_dict(a_log, ATTRIBUTION_LABELS),
                "per_snapshot_profile": [
                    {
                        "stage": PROFILE_LABELS[int(p_log[i].argmax().item())],
                        "distribution": {
                            PROFILE_LABELS[j]: round(F.softmax(p_log[i], dim=-1)[j].item(), 4)
                            for j in range(len(PROFILE_LABELS))
                        },
                    }
                    for i in range(p_log.size(0))
                ],
                "next_stage_prediction": [
                    {
                        "predicted_next_stage": PROFILE_LABELS[int(n_log[i].argmax().item())],
                        "distribution": {
                            PROFILE_LABELS[j]: round(F.softmax(n_log[i], dim=-1)[j].item(), 4)
                            for j in range(len(PROFILE_LABELS))
                        },
                    }
                    for i in range(n_log.size(0))
                ],
            }
        except Exception as e:
            neural_block = {"available": False, "error": str(e)}

    stage_counts = Counter(h["stage"] for h in heur_profile)
    dominant = stage_counts.most_common(1)[0][0] if heur_profile else "benign"

    report: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "status": "success",
        "module": "threat_attribution_profiling",
        "engine_notes": "Heterogeneous Graph Transformer + temporal Transformer; heuristics always included.",
        "event_count": len(events),
        "snapshot_count": len(snapshots),
        "campaign_attribution": {
            "primary_actor_class": att_label,
            "confidence": round(att_conf, 4),
            "distribution": att_dist,
            "source": "heuristic_rules",
        },
        "attack_profile": {
            "dominant_stage": dominant,
            "per_snapshot": [
                {
                    "time_start": datetime.utcfromtimestamp(float(snap.t_start)).isoformat() + "Z",
                    "time_end": datetime.utcfromtimestamp(float(snap.t_end)).isoformat() + "Z",
                    "heuristic_stage": heur_profile[i]["stage"],
                    "heuristic_confidence": heur_profile[i]["confidence"],
                }
                for i, snap in enumerate(snapshots)
            ],
        },
        "next_stage": {
            "per_snapshot": [
                {**next_heur[i], "window_index": i} for i in range(len(next_heur))
            ]
        },
        "neural_model": neural_block,
    }

    if neural_block.get("available") and "attribution_distribution" in neural_block:
        report["campaign_attribution"]["neural_distribution"] = neural_block["attribution_distribution"]

    return report


def run_demo(seed: int = 99, device: str = "cpu") -> Dict[str, Any]:
    """Synthetic campaign with coarser time windows so the HTTP demo stays responsive."""
    gen = SyntheticAttackGenerator(seed=seed)
    events, _labels = gen.generate_attack_campaign(
        n_events=400,
        attack_ratio=0.35,
        t_start=1_700_000_000.0,
        campaign_duration_sec=3600,
    )
    ckpt = os.path.join(_TA, "checkpoints", "best_hgt_seq.pt")
    return run_attribution_pipeline(
        events,
        checkpoint=ckpt if os.path.isfile(ckpt) else None,
        device=device,
        window_size_sec=600,
        slide_step_sec=180,
    )


def save_report(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
