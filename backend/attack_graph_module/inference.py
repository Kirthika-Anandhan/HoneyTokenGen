"""
Inference script — loads a trained TGN model, processes live/batch events,
generates temporal attack graphs, and outputs threat intelligence reports.

Usage:
    python inference.py --events events.json --checkpoint checkpoints/best_tgn.pt
    python inference.py --demo     # runs with synthetic attack data
"""

import os
import json
import torch
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict

from data.event_processor import (
    SyntheticAttackGenerator, TemporalGraphBuilder
)
from models.tgnn import AttackBehaviourTGN, BEHAVIOUR_LABELS
from graph.visualizer import (
    visualize_attack_campaign,
    visualize_anomaly_timeline,
    visualize_snapshot,
)


# ─────────────────────────────────────────────
# Threat report builder
# ─────────────────────────────────────────────

RISK_BANDS = [
    (0.0,  0.3,  'LOW',      '🟢'),
    (0.3,  0.6,  'MEDIUM',   '🟡'),
    (0.6,  0.8,  'HIGH',     '🟠'),
    (0.8,  1.01, 'CRITICAL', '🔴'),
]

def _risk_level(score: float) -> str:
    for lo, hi, label, icon in RISK_BANDS:
        if lo <= score < hi:
            return f'{icon} {label}'
    return '🔴 CRITICAL'


def build_threat_report(results: List[Dict]) -> Dict:
    """Summarise inference results into a structured threat intelligence report."""
    if not results:
        return {}

    max_anomaly    = max(r['anomaly_max']  for r in results)
    mean_anomaly   = np.mean([r['anomaly_mean'] for r in results])
    behaviours     = [r['behaviour'] for r in results]
    dominant_bhvr  = max(set(behaviours), key=behaviours.count)
    attack_windows = [r for r in results if r['anomaly_mean'] > 0.5]

    # Reconstruct campaign timeline
    timeline = []
    prev_bhvr = None
    for r in results:
        if r['behaviour'] != prev_bhvr or (timeline and r['anomaly_mean'] > 0.7):
            timeline.append({
                'time':       datetime.utcfromtimestamp(r['t_start']).isoformat(),
                'behaviour':  r['behaviour'],
                'confidence': f"{r['behaviour_conf']:.1%}",
                'anomaly':    f"{r['anomaly_mean']:.3f}",
            })
            prev_bhvr = r['behaviour']

    # Detect attack stages present
    stages_detected = list(dict.fromkeys(
        r['behaviour'] for r in results if r['behaviour'] != 'benign'
    ))

    report = {
        'generated_at':    datetime.utcnow().isoformat() + 'Z',
        'overall_risk':    _risk_level(max_anomaly),
        'max_anomaly':     round(max_anomaly, 4),
        'mean_anomaly':    round(float(mean_anomaly), 4),
        'dominant_stage':  dominant_bhvr,
        'attack_stages':   stages_detected,
        'alert_windows':   len(attack_windows),
        'total_snapshots': len(results),
        'timeline':        timeline,
        'recommendations': _recommendations(stages_detected, max_anomaly),
        'mitre_mapping':   _mitre_map(stages_detected),
    }
    return report


def _recommendations(stages: List[str], max_score: float) -> List[str]:
    recs = []
    if 'reconnaissance' in stages:
        recs.append('Block/rate-limit port-scanning sources immediately')
        recs.append('Enable honeypot responses on unused ports')
    if 'lateral_movement' in stages:
        recs.append('Isolate affected subnet segments')
        recs.append('Revoke and rotate credentials for lateral-move accounts')
    if 'privilege_escalation' in stages:
        recs.append('Audit sudo/SUID binaries on affected hosts')
        recs.append('Alert CISO — active privilege escalation detected')
    if 'exfiltration' in stages:
        recs.append('URGENT: Block outbound connections from affected IPs')
        recs.append('Engage incident response team immediately')
    if max_score > 0.8:
        recs.insert(0, '⚠️  Critical threat level — escalate to SOC immediately')
    return recs


def _mitre_map(stages: List[str]) -> Dict[str, str]:
    MITRE = {
        'reconnaissance':       'TA0043 — Reconnaissance / T1595 Active Scanning',
        'lateral_movement':     'TA0008 — Lateral Movement / T1021 Remote Services',
        'privilege_escalation': 'TA0004 — Privilege Escalation / T1548 Abuse Elevation Control',
        'exfiltration':         'TA0010 — Exfiltration / T1041 Exfiltration Over C2',
    }
    return {s: MITRE[s] for s in stages if s in MITRE}


# ─────────────────────────────────────────────
# Main inference pipeline
# ─────────────────────────────────────────────

def run_inference(
    events:      List[Dict],
    checkpoint:  str = None,
    device:      str = 'cpu',
    save_dir:    str = 'output',
    window_sec:  int = 300,
    slide_sec:   int = 60,
) -> Dict:
    """
    Full inference pipeline:
      1. Build temporal graph snapshots from events
      2. Run TGN to get anomaly scores + behaviour predictions
      3. Generate attack graph visualisations
      4. Build threat intelligence report

    Returns the full threat report dict.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f'[Inference] Processing {len(events)} events...')
    builder   = TemporalGraphBuilder(window_sec, slide_sec)
    snapshots = builder.build_snapshots(events)
    print(f'[Inference] Built {len(snapshots)} temporal snapshots.')

    # Load model
    model = AttackBehaviourTGN()
    if checkpoint and os.path.isfile(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f'[Inference] Loaded checkpoint from {checkpoint} (epoch {ckpt.get("epoch", "?")})')
    else:
        print('[Inference] No checkpoint found — using untrained model (for demo).')

    model = model.to(device)

    # Run TGN inference
    print('[Inference] Running TGN inference over snapshot sequence...')
    results = model.infer_sequence(snapshots, device=device)

    # Visualise
    print('[Inference] Generating attack graph visualisations...')
    graph_dir = os.path.join(save_dir, 'graphs')
    visualize_attack_campaign(snapshots, results, save_dir=graph_dir, max_snapshots=6)
    visualize_anomaly_timeline(results, save_path=os.path.join(graph_dir, 'anomaly_timeline.png'))

    # Most suspicious snapshot — full detail render
    if results:
        peak_idx  = max(range(len(results)), key=lambda i: results[i]['anomaly_max'])
        peak_snap = snapshots[peak_idx]
        peak_scores = {
            k: v.flatten()
            for k, v in results[peak_idx]['node_scores'].items()
        }
        visualize_snapshot(
            peak_snap, peak_scores,
            title=f'Peak Attack Snapshot (T{peak_idx+1}) — {results[peak_idx]["behaviour"]}',
            save_path=os.path.join(graph_dir, 'peak_snapshot.png'),
        )

    # Build and save threat report
    report = build_threat_report(results)
    report_path = os.path.join(save_dir, 'threat_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print()
    print('=' * 60)
    print('  THREAT INTELLIGENCE REPORT')
    print('=' * 60)
    print(f"  Risk level:      {report.get('overall_risk', 'N/A')}")
    print(f"  Max anomaly:     {report.get('max_anomaly', 0):.4f}")
    print(f"  Dominant stage:  {report.get('dominant_stage', 'N/A')}")
    print(f"  Attack stages:   {', '.join(report.get('attack_stages', []))}")
    print(f"  Alert windows:   {report.get('alert_windows', 0)}/{report.get('total_snapshots', 0)}")
    print()
    print('  MITRE ATT&CK mappings:')
    for stage, mapping in report.get('mitre_mapping', {}).items():
        print(f'    {stage}: {mapping}')
    print()
    print('  Recommendations:')
    for rec in report.get('recommendations', []):
        print(f'    • {rec}')
    print('=' * 60)
    print(f'  Report saved: {report_path}')
    print(f'  Graphs saved: {graph_dir}/')
    print('=' * 60)

    return report


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TGN Attack Graph Inference')
    parser.add_argument('--events',     type=str, default=None,
                        help='Path to events JSON file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_tgn.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir',   type=str, default='output')
    parser.add_argument('--demo',       action='store_true',
                        help='Run with synthetic attack data')
    parser.add_argument('--window_sec', type=int, default=300)
    parser.add_argument('--slide_sec',  type=int, default=60)
    args = parser.parse_args()

    if args.demo or args.events is None:
        print('[Demo] Generating synthetic attack campaign...')
        gen = SyntheticAttackGenerator(seed=99)
        events, labels = gen.generate_attack_campaign(
            n_events=600,
            attack_ratio=0.35,
            t_start=1_700_000_000.0,
            campaign_duration_sec=3600,
        )
        attack_count = sum(labels)
        print(f'[Demo] Generated {len(events)} events ({attack_count} attack, {len(events)-attack_count} benign)')
    else:
        print(f'[Inference] Loading events from {args.events}')
        with open(args.events) as f:
            events = json.load(f)

    run_inference(
        events,
        checkpoint=args.checkpoint,
        device=args.device,
        save_dir=args.save_dir,
        window_sec=args.window_sec,
        slide_sec=args.slide_sec,
    )