"""
FastAPI routes for Module 4 — Threat attribution & attack profiling (HGT + seq. Transformer).
"""

import json
import os
import traceback

from fastapi import APIRouter, HTTPException

from threat_attribution_module.inference import run_demo, save_report

MODULE_DIR = os.path.join(os.path.dirname(__file__), "threat_attribution_module")
OUTPUT_DIR = os.path.join(MODULE_DIR, "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "attribution_report.json")

router = APIRouter(prefix="/api/threat-attribution", tags=["Threat Attribution & Profiling"])


@router.get("/demo")
async def run_attribution_demo():
    try:
        report = run_demo(seed=99, device="cpu")
        save_report(report, REPORT_PATH)
        return {"status": "success", "data": report}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def get_last_attribution_report():
    try:
        if not os.path.exists(REPORT_PATH):
            raise HTTPException(
                status_code=404,
                detail="No report found. Run /api/threat-attribution/demo first.",
            )
        with open(REPORT_PATH, encoding="utf-8") as f:
            return {"status": "success", "data": json.load(f)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def attribution_health():
    ckpt = os.path.join(MODULE_DIR, "checkpoints", "best_hgt_seq.pt")
    return {
        "module": "threat_attribution_profiling",
        "neural_checkpoint_present": os.path.isfile(ckpt),
        "checkpoint_path": ckpt,
    }
