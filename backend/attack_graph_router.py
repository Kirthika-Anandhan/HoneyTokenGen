import os
import sys
import json
import base64
import traceback
from fastapi import APIRouter, HTTPException

MODULE_DIR = os.path.join(os.path.dirname(__file__), 'attack_graph_module')
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from data.event_processor import SyntheticAttackGenerator, TemporalGraphBuilder
from models.tgnn import AttackBehaviourTGN
from models.heads import BEHAVIOUR_LABELS
from inference import run_inference

router = APIRouter(prefix="/api/attack-graph", tags=["Attack Graph"])

OUTPUT_DIR = os.path.join(MODULE_DIR, 'output')
CHECKPOINT = os.path.join(MODULE_DIR, 'checkpoints', 'best_tgn.pt')
GRAPH_DIR  = os.path.join(OUTPUT_DIR, 'graphs')

def _img_to_base64(path: str) -> str:
    try:
        with open(path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        return f'data:image/png;base64,{data}'
    except Exception:
        return ''

@router.get("/demo")
async def run_demo():
    try:
        gen = SyntheticAttackGenerator(seed=99)
        events, labels = gen.generate_attack_campaign(
            n_events=600, attack_ratio=0.35,
            t_start=1_700_000_000.0, campaign_duration_sec=3600,
        )
        report = run_inference(events, checkpoint=CHECKPOINT, device='cpu', save_dir=OUTPUT_DIR)
        report['images'] = {
            'campaign': _img_to_base64(os.path.join(GRAPH_DIR, 'campaign_evolution.png')),
            'timeline': _img_to_base64(os.path.join(GRAPH_DIR, 'anomaly_timeline.png')),
            'peak':     _img_to_base64(os.path.join(GRAPH_DIR, 'peak_snapshot.png')),
        }
        report['event_count']  = len(events)
        report['attack_count'] = sum(labels)
        report['benign_count'] = len(labels) - sum(labels)
        return {"status": "success", "data": report}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report")
async def get_last_report():
    try:
        report_path = os.path.join(OUTPUT_DIR, 'threat_report.json')
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="No report found. Run /demo first.")
        with open(report_path) as f:
            report = json.load(f)
        report['images'] = {
            'campaign': _img_to_base64(os.path.join(GRAPH_DIR, 'campaign_evolution.png')),
            'timeline': _img_to_base64(os.path.join(GRAPH_DIR, 'anomaly_timeline.png')),
            'peak':     _img_to_base64(os.path.join(GRAPH_DIR, 'peak_snapshot.png')),
        }
        return {"status": "success", "data": report}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
