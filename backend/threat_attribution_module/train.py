"""
Train HGT + temporal Transformer on synthetic campaigns (weak labels from heuristics).
Run from backend directory:  python -m threat_attribution_module.train
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

_AG = os.path.join(BACKEND, "attack_graph_module")
if _AG not in sys.path:
    sys.path.insert(0, _AG)

from data.event_processor import SyntheticAttackGenerator, TemporalGraphBuilder  # noqa: E402

from threat_attribution_module.inference import (  # noqa: E402
    _campaign_attribution_heuristic,
    _events_in_snapshot,
    _profile_from_events,
)
from threat_attribution_module.models.hgt_sequence_model import (  # noqa: E402
    ThreatAttributionSequenceModel,
    ATTRIBUTION_LABELS,
    PROFILE_LABELS,
)

PROFILE_TO_IDX = {v: k for k, v in PROFILE_LABELS.items()}
ATTR_TO_IDX = {v: k for k, v in ATTRIBUTION_LABELS.items()}


def _labels_for_sequence(events, snapshots):
    per = [_events_in_snapshot(events, float(s.t_start), float(s.t_end)) for s in snapshots]
    prof = [PROFILE_TO_IDX.get(_profile_from_events(evs)[0], 0) for evs in per]
    att_s, _, _ = _campaign_attribution_heuristic(events)
    att = ATTR_TO_IDX.get(att_s, 0)
    next_t = prof[1:] + [prof[-1]]
    return att, prof, next_t


def main(epochs: int = 5, device: str = "cpu"):
    dev = torch.device(device)
    builder = TemporalGraphBuilder(600, 180)
    model = ThreatAttributionSequenceModel().to(dev)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.join(ckpt_dir, "best_hgt_seq.pt")

    model.train()
    for ep in range(epochs):
        total = 0.0
        n_batches = 0
        for seed in range(4):
            gen = SyntheticAttackGenerator(seed=42 + ep * 4 + seed)
            events, _ = gen.generate_attack_campaign(
                n_events=400, attack_ratio=0.35, t_start=1_700_000_000.0, campaign_duration_sec=3600
            )
            snaps = builder.build_snapshots(events)
            if len(snaps) < 2:
                continue
            att_idx, prof_idx, next_idx = _labels_for_sequence(events, snaps)
            opt.zero_grad()
            a_log, p_log, n_log = model(snaps, device=dev)
            loss_a = F.cross_entropy(a_log, torch.tensor([att_idx], device=dev))
            loss_p = F.cross_entropy(p_log, torch.tensor(prof_idx, device=dev))
            loss_n = F.cross_entropy(n_log, torch.tensor(next_idx, device=dev))
            loss = loss_a + loss_p + loss_n
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        if n_batches:
            print(f"epoch {ep+1}/{epochs} loss {total / n_batches:.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
