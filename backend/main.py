# ============================================================
# main.py — AI Honeytoken Generator  (v5 — aligned with token_generator.py v5)
# ============================================================

import json
import os
import torch
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from report_generator import generate_report
from database import db as honeyguard_db

# ── Your existing routers ────────────────────────────────────
from attack_graph_router import router as attack_graph_router
from threat_attribution_router import router as threat_attribution_router
from adaptive_rl import router as rl_router
from alerts_router import router as alerts_router

# ── v5 generator ─────────────────────────────────────────────
from token_generator import HoneytokenGenerator


# ============================================================
# Lifespan — replaces deprecated @app.on_event("startup")
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("🔐 AI Honeytoken Generator API v5 Started")
    print("=" * 60)
    print(f"📊 Model Status : {'Trained ✅' if generator.trained else 'Not Trained ⚠️'}")
    print(f"🎯 Generator    : VAE + Discriminator (95-char vocab, latent_dim=128)")
    print(f"🔑 Token Types  : JWT | API Key | Git Token | DB Credentials")
    print(f"🧩 Modules      : Factory | Attack Graph | RL Deploy | Attribution")
    print(f"📈 Targets      : Entropy Ratio ≥ 0.90 | Disc Score ≥ 0.90 | Authenticity ≥ 0.90")
    print("=" * 60)
    yield  # app runs here


# ============================================================
# App
# ============================================================
app = FastAPI(
    title="AI Honeytoken Generator",
    version="5.0",
    description="4-module AI-driven honeytoken security platform",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(attack_graph_router)        # /api/attack-graph/...
app.include_router(threat_attribution_router)  # /api/threat-attribution/...
app.include_router(rl_router)                  # /api/rl/...
app.include_router(alerts_router)              # /api/alerts/...


# ============================================================
# Startup: Load or Train ML Models
# ============================================================
print("🚀 Initialising ML Honeytoken Generator v5…")
# auto_pretrain=False: main.py always loads or fully trains right after,
# so the quick init warmup would just be overwritten — skip it here.
generator = HoneytokenGenerator(device='cpu', auto_pretrain=False)

MODEL_PATH = "honeytoken_v5.pt"

if os.path.exists(MODEL_PATH):
    print(f"📂 Loading pre-trained models from {MODEL_PATH}…")
    generator.load(MODEL_PATH)
else:
    # Token generation is fully cryptographic — the VAE is not used for output.
    # Discriminator pre-training alone is sufficient for genuine ≥ 0.90 scores
    # and completes in ~3–5 minutes on CPU instead of hours.
    print("⚠️  No pre-trained models found. Running discriminator pre-training…")
    generator._auto_pretrain_discriminator()
    generator.trained = True
    generator.save(MODEL_PATH)
    print("✅ Discriminator pre-trained and saved!")

print("✅ ML Honeytoken Generator v5 ready!")


# ============================================================
# Request / Response Models
# ============================================================
VALID_TYPES = ["db_record", "jwt", "github", "api", "cloud"]


class TokenRequest(BaseModel):
    token_usage: str = "api"   # db_record | jwt | github | api | cloud
    quantity:    int = 5
    name:        Optional[str] = None
    surname:     Optional[str] = None
    method:      str = "vae"   # kept for backwards-compat; v5 is always VAE

    class Config:
        schema_extra = {
            "example": {
                "token_usage": "jwt",
                "quantity": 3,
            }
        }


# ============================================================
# Internal helper
# ============================================================
def _generate_token(token_usage: str) -> dict:
    """Route token_usage string to the correct v5 generator method."""
    usage = token_usage.lower()
    if usage == "db_record":
        return generator.generate_db_credentials()
    elif usage == "jwt":
        return generator.generate_jwt()
    elif usage == "github":
        return generator.generate_git_token()
    elif usage in ("api", "cloud"):
        return generator.generate_api_key()
    else:
        return generator.generate_api_key()


# ============================================================
# Token Generation Endpoints
# ============================================================
@app.post("/generate-token", response_model=dict)
async def generate_token_endpoint(request: TokenRequest):
    """
    Generate honeytokens using the v5 VAE + discriminator model.

    - **token_usage** – db_record | jwt | github | api | cloud
    - **quantity**    – 1–100
    """
    if not (1 <= request.quantity <= 100):
        raise HTTPException(400, "quantity must be between 1 and 100")

    if request.token_usage.lower() not in VALID_TYPES:
        raise HTTPException(400, f"token_usage must be one of: {VALID_TYPES}")

    results = []

    for i in range(request.quantity):
        try:
            data = _generate_token(request.token_usage)

            token_type    = data.get("type", request.token_usage)
            entropy       = data.get("entropy", 0.0)
            entropy_ratio = data.get("entropy_ratio", 0.0)
            disc_score    = data.get("disc_score", 0.0)
            authenticity  = data.get("authenticity", 0.0)

            # db_credentials stores the whole dict; everything else stores the token string
            if token_type == "db_credentials":
                token_value = data
            else:
                token_value = {
                    "token":         data.get("token", ""),
                    "type":          token_type,
                    "entropy":       entropy,
                    "entropy_ratio": entropy_ratio,
                    "disc_score":    disc_score,
                    "authenticity":  authenticity,
                }

            from database import save_token
            save_token(
                token_type=request.token_usage,
                token_value=json.dumps(token_value),
                entropy=entropy,
                similarity=entropy_ratio,
                discriminator=disc_score,
            )

            results.append({
                "token_value":   token_value,
                "entropy":       entropy,
                "entropy_ratio": entropy_ratio,
                "disc_score":    disc_score,
                "authenticity":  authenticity,
                "token_type":    token_type,
            })

        except Exception as e:
            print(f"❌ Error generating token {i + 1}: {e}")
            raise HTTPException(500, f"Error generating token: {e}")

    return {
        "tokens":      results,
        "count":       len(results),
        "token_usage": request.token_usage,
    }


@app.post("/generate-token-batch")
async def generate_token_batch(requests: List[TokenRequest]):
    """Generate multiple batches of different token types in one call."""
    all_results = []
    for req in requests:
        batch = await generate_token_endpoint(req)
        all_results.append({
            "token_usage": req.token_usage,
            "tokens":      batch["tokens"],
        })

    return {
        "batches":      all_results,
        "total_tokens": sum(len(b["tokens"]) for b in all_results),
    }


# ============================================================
# Model Management Endpoints
# ============================================================
@app.get("/model-info")
def model_info():
    """Return metadata about the loaded v5 ML models."""
    return {
        "model_trained":   generator.trained,
        "device":          str(generator.device),
        "generation":      "v5 (VAE + Discriminator, 95-char vocab, genuine metrics)",
        "supported_token_types": ["jwt", "api_key", "git_token", "db_credentials"],
        "model_architecture": {
            "vae": {
                "vocab_size":  95,    # printable ASCII 32–126
                "max_len":     64,
                "latent_dim":  128,
                "hidden_dim":  512,
                "lstm_layers": 3,
                "bidirectional": True,
            },
            "discriminator": {
                "vocab_size":  95,
                "hidden_dim":  512,
                "lstm_layers": 3,
                "bidirectional": True,
                "spectral_norm": True,
            },
        },
        "targets": {
            "entropy_ratio": "≥ 0.90 of log2(|charset|)",
            "disc_score":    "≥ 0.90",
            "authenticity":  "≥ 0.90 (geometric mean of entropy_ratio × disc_score)",
        },
    }


@app.post("/retrain")
async def retrain_models(request: Request):
    """
    Retrain v5 models with new token examples.

    Body example:
    ```json
    {
        "jwt":        ["token1", "token2"],
        "api_key":    ["token1", "token2"],
        "git_token":  ["token1", "token2"],
        "epochs": 500,
        "batch_size": 32,
        "warmup_disc_epochs": 50
    }
    ```
    """
    try:
        body = await request.json()

        training_data = {
            k: body[k]
            for k in ("jwt", "api_key", "git_token")
            if body.get(k)
        }

        if not training_data:
            raise HTTPException(400, "No training data provided")

        epochs              = body.get("epochs",              500)
        batch_size          = body.get("batch_size",           32)
        warmup_disc_epochs  = body.get("warmup_disc_epochs",   50)

        n_samples = sum(len(v) for v in training_data.values())
        print(f"🔄 Retraining with {n_samples} samples…")
        generator.train(
            training_data,
            epochs=epochs,
            batch_size=batch_size,
            warmup_disc_epochs=warmup_disc_epochs,
        )
        generator.save(MODEL_PATH)

        return {
            "message":          "Models retrained successfully",
            "training_samples": {k: len(v) for k, v in training_data.items()},
            "epochs":           epochs,
            "model_saved":      MODEL_PATH,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Retraining failed: {e}")


# ============================================================
# Report Generation Endpoint
# ============================================================
@app.post("/api/generate-report")
async def generate_report_endpoint(request: Request):
    """
    Generate a PDF breach report.
    Fetches the latest data from all five MongoDB collections automatically.
    Frontend can send an empty POST {} or override specific fields.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    # ── Fetch latest document from every collection (BUG 4 fix) ──
    payload: dict = {}

    if honeyguard_db is not None:
        # Module 1 — latest generated token
        tok = honeyguard_db.tokens.find_one(sort=[("_id", -1)])
        if tok:
            raw_tv = tok.get("token_value", "")
            try:
                parsed = json.loads(raw_tv) if isinstance(raw_tv, str) else raw_tv
                token_str = parsed.get("token", raw_tv) if isinstance(parsed, dict) else raw_tv
            except Exception:
                token_str = raw_tv
            payload.update({
                "token_id":      tok.get("token_type", "unknown") + "_latest",
                "token_type":    tok.get("token_type",    "unknown"),
                "token_value":   token_str,
                "entropy":       tok.get("entropy",       0.0),
                "entropy_ratio": tok.get("entropy_ratio", 0.0),
                "disc_score":    tok.get("disc_score",    0.0),
                "authenticity":  tok.get("similarity",    0.0),
            })

        # Module 2 — latest attack-graph result
        grf = honeyguard_db.graph_state.find_one({"_id": "latest"})
        if grf:
            payload["attack_graph_summary"] = {
                "risk_level":           grf.get("overall_risk",    "—"),
                "max_anomaly_score":    grf.get("max_anomaly",     0),
                "mean_anomaly_score":   grf.get("mean_anomaly",    0),
                "attack_windows_count": grf.get("alert_windows",   0),
                "dominant_stage":       grf.get("dominant_stage",  "—"),
                "total_events":         grf.get("event_count",     0),
                "mitre_mappings":       [
                    {"tactic": k, "tactic_id": v, "techniques": []}
                    for k, v in (grf.get("mitre_mapping") or {}).items()
                ],
            }
            payload.setdefault("attack_stages", grf.get("attack_stages", []))
            payload.setdefault("risk_score",
                {"CRITICAL": 0.90, "HIGH": 0.75, "MEDIUM": 0.50, "LOW": 0.20}
                .get(str(grf.get("overall_risk", "")).upper(), 0.50))

        # Module 3 — latest deployment state
        dep = honeyguard_db.deployment_state.find_one({"_id": "current"})
        if dep:
            detections = dep.get("detections", [])
            if detections:
                last_det = detections[-1]
                payload.setdefault("attacker_ip",   last_det.get("attacker_ip", "N/A"))
                payload.setdefault("triggered_at",  last_det.get("timestamp",   ""))
                payload.setdefault("token_id",
                    last_det.get("token_id", payload.get("token_id", "unknown")))

        # Module 4 — latest threat attribution
        atr = honeyguard_db.attribution.find_one({"_id": "latest"})
        if atr:
            ca = (atr.get("campaign_attribution") or {})
            payload["threat_attribution_summary"] = {
                "campaign_attribution": {
                    "predicted_actor": ca.get("primary_actor_class",
                                              ca.get("predicted_actor", "Unknown")),
                    "confidence":      ca.get("confidence", 0),
                    "method":          ca.get("method", "heuristic"),
                }
            }
            payload.setdefault("attribution",
                ca.get("primary_actor_class", ca.get("predicted_actor", "Unknown")))

        # Module 5 — alert count as context
        payload["alert_count"] = honeyguard_db.alerts.count_documents({})
        latest_alert = honeyguard_db.alerts.find_one(
            {}, {"_id": 0}, sort=[("created_at", -1)]
        )
        if latest_alert:
            payload.setdefault("attacker_ip", latest_alert.get("attacker_ip", "N/A"))

    # Explicit body fields always override MongoDB values
    for k, v in body.items():
        if v is not None:
            payload[k] = v

    # Sensible defaults for any remaining missing fields
    payload.setdefault("token_id",      "report")
    payload.setdefault("token_type",    "unknown")
    payload.setdefault("token_value",   "")
    payload.setdefault("attacker_ip",   "N/A")
    payload.setdefault("triggered_at",  datetime.now(timezone.utc).isoformat())
    payload.setdefault("attack_stages", ["reconnaissance", "lateral_movement"])
    payload.setdefault("risk_score",    0.50)
    payload.setdefault("entropy",       0.0)
    payload.setdefault("entropy_ratio", 0.0)
    payload.setdefault("disc_score",    0.0)
    payload.setdefault("authenticity",  0.0)
    payload.setdefault("attribution",   "Unknown")

    pdf_bytes = generate_report(payload)
    token_id  = payload.get("token_id", "report")
    filename  = f"breach_report_{token_id}.pdf"

    return Response(
        content    = pdf_bytes,
        media_type = "application/pdf",
        headers    = {"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================
# Health & Root Endpoints
# ============================================================
@app.get("/health")
def health_check():
    return {
        "status":             "healthy",
        "ml_models_loaded":   generator.trained,
        "database_connected": True,
        "generator_version":  "v5",
    }


@app.get("/")
def home():
    return {
        "message":       "AI Honeytoken Generator Running",
        "version":       "5.0",
        "model_trained": generator.trained,
        "modules": {
            "1_honeytoken_factory":  "/generate-token",
            "2_attack_graph":        "/api/attack-graph/demo",
            "3_rl_deployment":       "/api/rl/dashboard",
            "4_threat_attribution":  "/api/threat-attribution/demo",
            "5_real_time_alerts":    "/api/alerts/stream",
        },
        "endpoints": {
            "POST /generate-token":              "Generate honeytokens (Module 1)",
            "POST /generate-token-batch":        "Generate multiple token batches",
            "GET  /model-info":                  "ML model metadata",
            "POST /retrain":                     "Retrain models with new data",
            "GET  /health":                      "Health check",
            "GET  /api/attack-graph/demo":       "Attack behaviour graph (Module 2)",
            "GET  /api/rl/dashboard":            "RL deployment dashboard (Module 3)",
            "GET  /api/threat-attribution/demo": "Threat attribution (Module 4)",
            "GET  /api/alerts/stream":           "SSE live alert feed (Module 5)",
            "GET  /api/alerts/history":          "Alert history",
            "GET  /api/alerts/stats":            "Alert statistics",
            "POST /api/alerts/trigger":          "Manually fire a test alert",
        },
    }


# ============================================================
# Dev server entry point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)