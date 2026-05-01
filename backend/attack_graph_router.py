# ============================================================
# attack_graph_router.py — Module 2
# Writes results to MongoDB `graph_state` collection.
# Reads from MongoDB instead of JSON file (BUG 3 fix).
# ============================================================

import os
import sys
import json
import base64
import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from database import db

MODULE_DIR = os.path.join(os.path.dirname(__file__), "attack_graph_module")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from data.event_processor import SyntheticAttackGenerator  # noqa: E402
from inference import run_inference                         # noqa: E402

router     = APIRouter(prefix="/api/attack-graph", tags=["Attack Graph"])
OUTPUT_DIR = os.path.join(MODULE_DIR, "output")
CHECKPOINT = os.path.join(MODULE_DIR, "checkpoints", "best_tgn.pt")
GRAPH_DIR  = os.path.join(OUTPUT_DIR, "graphs")

# ── MongoDB collection ────────────────────────────────────────
_graph_col = db["graph_state"] if db is not None else None


def _img_to_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    except Exception:
        return ""


@router.get("/demo")
async def run_demo():
    try:
        gen    = SyntheticAttackGenerator(seed=99)
        events, labels = gen.generate_attack_campaign(
            n_events=600, attack_ratio=0.35,
            t_start=1_700_000_000.0, campaign_duration_sec=3600,
        )
        report = run_inference(
            events, checkpoint=CHECKPOINT, device="cpu", save_dir=OUTPUT_DIR
        )
        report["event_count"]  = len(events)
        report["attack_count"] = int(sum(labels))
        report["benign_count"] = int(len(labels) - sum(labels))

        # ── Persist to MongoDB (exclude large base64 images) ──
        if _graph_col is not None:
            try:
                # json round-trip converts numpy / non-serialisable types to Python natives
                doc = json.loads(json.dumps(report, default=str))
                doc["_id"]        = "latest"
                doc["updated_at"] = datetime.now(timezone.utc)
                _graph_col.replace_one({"_id": "latest"}, doc, upsert=True)
                print("✅ graph_state saved to MongoDB")
            except Exception as e:
                print(f"[AttackGraph] MongoDB write error: {e}")

        # Add images for the API response (NOT stored in MongoDB)
        report["images"] = {
            "campaign": _img_to_base64(os.path.join(GRAPH_DIR, "campaign_evolution.png")),
            "timeline": _img_to_base64(os.path.join(GRAPH_DIR, "anomaly_timeline.png")),
            "peak":     _img_to_base64(os.path.join(GRAPH_DIR, "peak_snapshot.png")),
        }
        return {"status": "success", "data": report}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def get_last_report():
    """Load last analysis from MongoDB (BUG 3 fix — no JSON file read)."""
    try:
        if _graph_col is None:
            raise HTTPException(500, "Database not connected")

        doc = _graph_col.find_one({"_id": "latest"}, {"_id": 0, "updated_at": 0})
        if doc is None:
            raise HTTPException(
                404, "No report found. Run /api/attack-graph/demo first."
            )

        # Re-attach images from disk (they are never stored in MongoDB)
        doc["images"] = {
            "campaign": _img_to_base64(os.path.join(GRAPH_DIR, "campaign_evolution.png")),
            "timeline": _img_to_base64(os.path.join(GRAPH_DIR, "anomaly_timeline.png")),
            "peak":     _img_to_base64(os.path.join(GRAPH_DIR, "peak_snapshot.png")),
        }
        return {"status": "success", "data": doc}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
