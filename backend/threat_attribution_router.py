# ============================================================
# threat_attribution_router.py — Module 4
# Writes attribution results to MongoDB `attribution` collection.
# Reads from MongoDB instead of JSON file (BUG 3 fix).
# Duplicate serve_frontend functions removed.
# ============================================================

import json
import os
import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from threat_attribution_module.inference import run_demo
from database import db

MODULE_DIR  = os.path.join(os.path.dirname(__file__), "threat_attribution_module")
OUTPUT_DIR  = os.path.join(MODULE_DIR, "output")

router = APIRouter(
    prefix="/api/threat-attribution",
    tags=["Threat Attribution & Profiling"],
)

# ── MongoDB collection ────────────────────────────────────────
_attr_col = db["attribution"] if db is not None else None


@router.get("/demo")
async def run_attribution_demo():
    """Run HGT + Transformer attribution pipeline and persist to MongoDB."""
    try:
        report = run_demo(seed=99, device="cpu")

        if _attr_col is not None:
            try:
                # json round-trip converts any non-serialisable types to strings
                doc = json.loads(json.dumps(report, default=str))
                doc["_id"]        = "latest"
                doc["updated_at"] = datetime.now(timezone.utc)
                _attr_col.replace_one({"_id": "latest"}, doc, upsert=True)
                print("✅ attribution saved to MongoDB")
            except Exception as e:
                print(f"[Attribution] MongoDB write error: {e}")

        return {"status": "success", "data": report}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def get_last_attribution_report():
    """Load last attribution result from MongoDB (BUG 3 fix — no JSON file read)."""
    try:
        if _attr_col is None:
            raise HTTPException(500, "Database not connected")

        doc = _attr_col.find_one({"_id": "latest"}, {"_id": 0, "updated_at": 0})
        if doc is None:
            raise HTTPException(
                404,
                "No report found. Run /api/threat-attribution/demo first.",
            )
        return {"status": "success", "data": doc}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def attribution_health():
    ckpt = os.path.join(MODULE_DIR, "checkpoints", "best_hgt_seq.pt")
    return {
        "module":                   "threat_attribution_profiling",
        "neural_checkpoint_present": os.path.isfile(ckpt),
        "checkpoint_path":           ckpt,
    }
