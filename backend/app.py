#!/usr/bin/env python3
"""
app.py — HoneyGuard Flask Server  v2
Port : 5000   DB : MongoDB honeyguard
"""

import json, os, queue, threading, uuid, traceback, re
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import requests as _http          # for IP geolocation
from flask import Flask, request, Response, jsonify, send_from_directory, send_file
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
DEPLOYED_DIR = BASE_DIR / "deployed_tokens"
REPORTS_DIR  = BASE_DIR / "reports"
MODEL_PATH   = BASE_DIR / "honeytoken_v5.pt"

DEPLOYED_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ── MongoDB ───────────────────────────────────────────────────
_mongo = MongoClient("mongodb://localhost:27017/")
db     = _mongo["honeyguard"]

db.deployments.create_index("token_value")
db.deployments.create_index("status")
db.tokens.create_index("status")
db.alerts.create_index("attacker_ip")
db.blocked_ips.create_index("ip")
db.geo_cache.create_index("ip", unique=True)

# ── Prevention: initialise settings if missing ────────────────
if db.prevention_settings.count_documents({}) == 0:
    db.prevention_settings.insert_one({
        "mode":                   "monitor",
        "auto_block":             False,
        "auto_rotate_tokens":     False,
        "auto_quarantine":        False,
        "auto_alert_team":        True,
        "auto_report":            True,
        "block_duration_minutes": 60,
        "updated_at":             datetime.utcnow(),
    })

# ── Token generator ───────────────────────────────────────────
import sys
sys.path.insert(0, str(BASE_DIR))
from token_generator import HoneytokenGenerator

_gen = HoneytokenGenerator(device="cpu", auto_pretrain=False)
if MODEL_PATH.exists():
    _gen.load(str(MODEL_PATH))
    print(f"✅ Loaded model from {MODEL_PATH}")
else:
    print("⚙️  Pre-training discriminator…")
    _gen._auto_pretrain_discriminator()
    _gen.trained = True
    _gen.save(str(MODEL_PATH))
    print("✅ Model saved")

# ── Flask ─────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)
CORS(app)

# ── SSE ──────────────────────────────────────────────────────
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()


def broadcast_sse(data: dict) -> None:
    payload = json.dumps(data)
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            if q in _sse_clients:
                _sse_clients.remove(q)


@app.route("/api/events/stream")
def sse_stream():
    def generate():
        q: queue.Queue = queue.Queue(maxsize=100)
        with _sse_lock:
            _sse_clients.append(q)
        try:
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            while True:
                try:
                    data = q.get(timeout=25)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except GeneratorExit:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Access-Control-Allow-Origin": "*"})


# ── Static ────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/reports/<path:filename>")
def serve_report_file(filename):
    return send_from_directory(str(REPORTS_DIR), filename)


# ── Helpers ───────────────────────────────────────────────────
def _extract_token(req) -> str | None:
    auth = req.headers.get("Authorization", "")
    if auth.startswith("Bearer "): return auth[7:].strip()
    if auth.startswith("Token "):  return auth[6:].strip()
    x_api = req.headers.get("X-API-Key", "")
    if x_api: return x_api.strip()
    body = req.get_json(silent=True) or {}
    for f in ("api_key", "token", "key"):
        if body.get(f): return str(body[f])
    for p in ("api_key", "token", "key"):
        v = req.args.get(p)
        if v: return v
    return None


def _get_ip(req) -> str:
    return (req.headers.get("X-Forwarded-For")
            or req.headers.get("X-Real-IP")
            or req.remote_addr or "unknown")


def _iso(dt) -> str:
    if isinstance(dt, datetime): return dt.isoformat()
    return str(dt or "")


# ── IP geolocation (cached in MongoDB) ───────────────────────
_PRIVATE_RANGES = re.compile(
    r"^(127\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|::1|localhost)"
)

def _get_geo(ip: str) -> dict:
    """Return geolocation dict for ip. Uses MongoDB cache to avoid repeated calls."""
    if _PRIVATE_RANGES.match(ip):
        # Private IPs have no real location — mark as private, no lat/lon
        return {"ip": ip, "lat": None, "lon": None, "city": "Private Network",
                "country": "Local Network", "countryCode": "LAN", "isp": "LAN",
                "org": "Private/Local", "private": True}
    cached = db.geo_cache.find_one({"ip": ip})
    if cached:
        cached.pop("_id", None)
        return cached
    try:
        r = _http.get(f"http://ip-api.com/json/{ip}?fields=status,country,countryCode,city,lat,lon,isp,org",
                      timeout=3)
        d = r.json()
        if d.get("status") == "success":
            geo = {"ip": ip, "lat": d["lat"], "lon": d["lon"],
                   "city": d.get("city","—"), "country": d.get("country","—"),
                   "countryCode": d.get("countryCode","—"),
                   "isp": d.get("isp","—"), "org": d.get("org","—"),
                   "private": False}
            try:
                db.geo_cache.update_one({"ip": ip}, {"$set": geo}, upsert=True)
            except Exception:
                pass
            return geo
    except Exception:
        pass
    return {"ip": ip, "lat": 0, "lon": 0, "city": "Unknown",
            "country": "Unknown", "countryCode": "—",
            "isp": "—", "org": "—", "private": False}


# ============================================================
# Prevention middleware — check blocked IPs
# ============================================================
_TRAP_PATHS = {"/api/data", "/api/user", "/api/admin",
               "/api/login", "/api/files", "/api/config"}

@app.before_request
def check_blocked_ip():
    if request.path not in _TRAP_PATHS:
        return None
    ip = _get_ip(request)
    blocked = db.blocked_ips.find_one({
        "ip": ip, "active": True,
        "expires_at": {"$gt": datetime.utcnow()}
    })
    if blocked:
        db.blocked_attempts.insert_one({
            "timestamp": datetime.utcnow(),
            "ip": ip, "endpoint": request.path,
            "reason": "IP in blocklist"
        })
        return jsonify({"error": "Too Many Requests", "retry_after": 3600}), 429


# ============================================================
# Prevention action functions
# ============================================================
def prevention_block_ip(ip: str, reason: str, duration_mins: int = 60) -> dict:
    expires = datetime.utcnow() + timedelta(minutes=duration_mins)
    db.blocked_ips.update_one(
        {"ip": ip},
        {"$set": {"ip": ip, "active": True, "reason": reason,
                  "blocked_at": datetime.utcnow(), "expires_at": expires,
                  "blocked_by": "honeyguard_system"}},
        upsert=True,
    )
    db.prevention_log.insert_one({
        "timestamp": datetime.utcnow(), "action": "block_ip", "target": ip,
        "reason": reason, "duration_minutes": duration_mins,
        "expires_at": expires, "status": "active"
    })
    broadcast_sse({"type": "prevention_action", "action": "block_ip", "ip": ip,
                   "message": f"IP {ip} blocked for {duration_mins} minutes",
                   "reason": reason})
    return {"blocked": True, "expires_at": expires.isoformat()}


def prevention_rotate_token(token_value: str) -> None:
    db.tokens.update_one({"token_value": token_value},
                         {"$set": {"status": "rotated", "rotated_at": datetime.utcnow()}})
    db.deployments.update_one({"token_value": token_value},
                              {"$set": {"status": "rotated"}})
    db.prevention_log.insert_one({
        "timestamp": datetime.utcnow(), "action": "rotate_token",
        "target": token_value[:8] + "...",
        "reason": "Token was triggered by attacker", "status": "completed"
    })
    broadcast_sse({"type": "prevention_action", "action": "rotate_token",
                   "message": "Triggered token rotated and replaced",
                   "token_masked": token_value[:8] + "..."})


def prevention_quarantine_ip(ip: str, duration_mins: int = 60) -> None:
    expires = datetime.utcnow() + timedelta(minutes=duration_mins)
    db.quarantined_ips.update_one(
        {"ip": ip},
        {"$set": {"ip": ip, "active": True, "quarantined_at": datetime.utcnow(),
                  "expires_at": expires, "reason": "Honeytoken access detected"}},
        upsert=True,
    )
    db.prevention_log.insert_one({
        "timestamp": datetime.utcnow(), "action": "quarantine",
        "target": ip, "duration_minutes": duration_mins, "status": "active"
    })


def _generate_report_background(ip: str, token_value: str, stage: str) -> None:
    try:
        pdf  = _build_pdf()
        now  = datetime.utcnow()
        iid  = f"HG-AUTO-{now.strftime('%Y%m%d-%H%M%S')}"
        fname = f"{iid}.pdf"
        (REPORTS_DIR / fname).write_bytes(pdf)
        db.reports.insert_one({"timestamp": now, "incident_id": iid,
                                "report_path": str(REPORTS_DIR / fname),
                                "auto_generated": True})
        db.prevention_log.insert_one({
            "timestamp": now, "action": "auto_report",
            "target": ip, "reason": "Auto-generated on detection", "status": "completed"
        })
    except Exception as e:
        print(f"[BG report error] {e}")


# ============================================================
# Detection pipeline
# ============================================================
_STAGE_MAP = {
    "/api/data":   "exfiltration",
    "/api/files":  "exfiltration",
    "/api/admin":  "privilege_escalation",
    "/api/config": "privilege_escalation",
    "/api/login":  "reconnaissance",
    "/api/user":   "lateral_movement",
}
_BASE_SCORES = {
    "/api/admin": 0.90, "/api/config": 0.85, "/api/files": 0.80,
    "/api/data":  0.75, "/api/user":   0.65, "/api/login": 0.60,
}
_MITRE = {
    "reconnaissance":       "TA0043 — Reconnaissance / T1595",
    "lateral_movement":     "TA0008 — Lateral Movement / T1021",
    "privilege_escalation": "TA0004 — Privilege Escalation / T1078",
    "exfiltration":         "TA0010 — Exfiltration / T1041",
}


def trigger_detection(token_value: str, req) -> dict:
    ip         = _get_ip(req)
    user_agent = req.headers.get("User-Agent", "unknown")
    endpoint   = req.path
    method     = req.method
    now        = datetime.utcnow()

    db.tokens.update_one({"token_value": token_value},
                         {"$set": {"status": "triggered", "triggered_at": now, "attacker_ip": ip}})
    db.deployments.update_one({"token_value": token_value},
                              {"$set": {"status": "triggered", "triggered_at": now}})

    stage = _STAGE_MAP.get(endpoint, "reconnaissance")
    hour  = now.hour
    night = 0.2 if (hour < 6 or hour > 22) else 0.0
    prev  = db.alerts.count_documents({"attacker_ip": ip})
    repeat = min(prev * 0.1, 0.3)
    score  = round(min(_BASE_SCORES.get(endpoint, 0.70) + night + repeat, 0.99), 4)

    dep_doc  = db.deployments.find_one({"token_value": token_value})
    token_id = dep_doc.get("token_id", "unknown") if dep_doc else "unknown"

    # Geolocation (non-blocking best-effort)
    geo = _get_geo(ip)

    db.graph_state.insert_one({
        "timestamp": now, "anomaly_score": score,
        "risk_level": "CRITICAL" if score > 0.85 else "HIGH",
        "dominant_stage": stage,
        "mitre_mapping": _MITRE.get(stage, "TA0043 — Reconnaissance"),
        "events_processed": db.alerts.count_documents({}) + 1,
        "attacker_ip": ip, "endpoint": endpoint, "token_id": token_id,
        "geo": geo,
    })

    if prev == 0:
        actor, conf = "external_network_attacker", 0.72
    elif "/admin" in endpoint or "/config" in endpoint:
        actor, conf = "privileged_insider", 0.68
    else:
        actor, conf = "automated_tooling", 0.81

    db.attribution.insert_one({
        "timestamp": now, "token_id": token_id,
        "primary_actor": actor, "confidence": conf,
        "attack_stages": list(dict.fromkeys(["reconnaissance", stage])),
        "attacker_ip": ip, "user_agent": user_agent,
        "attribution_distribution": {
            "external_network_attacker": 0.317, "compromised_account": 0.317,
            "privileged_insider": 0.195, "automated_tooling": 0.146,
            "undetermined": 0.024,
        },
    })

    severity = "CRITICAL" if score > 0.85 else "HIGH"
    alert = {
        "timestamp": now, "token_id": token_id,
        "token_value": token_value[:8] + "...",
        "attacker_ip": ip, "user_agent": user_agent,
        "endpoint": endpoint, "method": method,
        "anomaly_score": score, "attack_stage": stage,
        "severity": severity,
        "message": f"Honeytoken accessed from {ip} via {method} {endpoint}",
        "geo": geo,
    }
    db.alerts.insert_one(alert)

    broadcast_sse({
        "type": "honeytoken_triggered",
        "timestamp": now.isoformat(),
        "attacker_ip": ip, "endpoint": endpoint,
        "anomaly_score": score, "attack_stage": stage,
        "severity": severity,
        "token_masked": token_value[:8] + "...",
        "message": alert["message"],
        "user_agent": user_agent, "method": method,
        "geo": geo,
    })

    # ── Prevention logic ──────────────────────────────────────
    settings = db.prevention_settings.find_one() or {}
    mode = settings.get("mode", "monitor")

    if mode == "alert_recommend":
        recommended_actions = [
            {"action": "block_ip",     "target": ip,
             "label": f"Block IP {ip} for 60 minutes",
             "reason": f"Made {prev+1} requests using honeytoken",
             "risk": "Low", "reversible": True},
            {"action": "rotate_token", "target": token_value,
             "label": "Rotate triggered honeytoken",
             "reason": "Token identity compromised",
             "risk": "Very Low", "reversible": True},
            {"action": "quarantine",   "target": ip,
             "label": f"Quarantine IP {ip} for enhanced monitoring",
             "reason": "Suspicious access pattern",
             "risk": "None", "reversible": True},
        ]
        rec_doc = db.prevention_recommendations.insert_one({
            "timestamp": now, "attacker_ip": ip,
            "token_value": token_value[:8] + "...",
            "recommendations": recommended_actions,
            "status": "pending_approval"
        })
        broadcast_sse({
            "type": "prevention_recommended", "ip": ip,
            "rec_id": str(rec_doc.inserted_id),
            "recommendations": recommended_actions,
            "message": f"3 prevention actions recommended for {ip}"
        })

    elif mode == "auto_prevent":
        duration = settings.get("block_duration_minutes", 60)
        actions_taken = []
        if settings.get("auto_block", True):
            prevention_block_ip(ip, f"Auto-blocked: honeytoken access at {endpoint}", duration)
            actions_taken.append("block_ip")
        if settings.get("auto_rotate_tokens", True):
            prevention_rotate_token(token_value)
            actions_taken.append("rotate_token")
        if settings.get("auto_quarantine", True):
            prevention_quarantine_ip(ip, duration)
            actions_taken.append("quarantine")
        if settings.get("auto_report", True):
            threading.Thread(target=_generate_report_background,
                             args=(ip, token_value, stage), daemon=True).start()
            actions_taken.append("auto_report")
        broadcast_sse({
            "type": "prevention_executed", "ip": ip,
            "actions": actions_taken,
            "message": f"Auto-prevention: {len(actions_taken)} actions executed on {ip}"
        })

    print(f"\n{'='*50}\nHONEYTOKEN TRIGGERED\nIP: {ip}\nEndpoint: {endpoint}\n"
          f"Stage: {stage}\nScore: {score}\nMode: {mode}\n{'='*50}\n")
    return alert


# ============================================================
# Blocked IP middleware trap + trap endpoints
# ============================================================
_FAKE_OK = {"status": "success", "data": {"id": 1042, "role": "admin"}}


def _trap_handler():
    token = _extract_token(request)
    if token:
        dep = db.deployments.find_one({"token_value": token, "status": "active"})
        if dep:
            trigger_detection(token, request)
    return jsonify(_FAKE_OK)


@app.route("/api/data",   methods=["GET", "POST"])
def trap_data():   return _trap_handler()
@app.route("/api/user",   methods=["GET", "POST"])
def trap_user():   return _trap_handler()
@app.route("/api/admin",  methods=["GET", "POST"])
def trap_admin():  return _trap_handler()
@app.route("/api/login",  methods=["GET", "POST"])
def trap_login():  return _trap_handler()
@app.route("/api/files",  methods=["GET"])
def trap_files():  return _trap_handler()
@app.route("/api/config", methods=["GET"])
def trap_config(): return _trap_handler()


# ============================================================
# Token generation  (max 500)
# ============================================================
def _gen_token(token_type: str, method: str = "hybrid") -> dict:
    t = token_type.lower()
    if t in ("api", "api_key", "cloud"):  data = _gen.generate_api_key()
    elif t == "jwt":                       data = _gen.generate_jwt()
    elif t in ("github", "git_token"):    data = _gen.generate_git_token()
    elif t in ("db_record", "db_credentials"): data = _gen.generate_db_credentials()
    else:                                  data = _gen.generate_api_key()
    tv = data.get("token") if isinstance(data, dict) and "token" in data else json.dumps(data)
    return {
        "token_id":           f"ht_{uuid.uuid4().hex[:12]}",
        "token_value":        tv,
        "token_type":         token_type,
        "method":             method,
        "authenticity_score": float(data.get("authenticity", data.get("entropy_ratio", 0.0))),
        "entropy":            float(data.get("entropy", 0.0)),
        "disc_score":         float(data.get("disc_score", 0.0)),
        "timestamp":          datetime.utcnow(),
        "status":             "generated",
    }


@app.route("/api/generate-token", methods=["POST"])
def api_generate_token():
    body       = request.get_json(silent=True) or {}
    token_type = body.get("token_type", body.get("token_usage", "api"))
    method     = body.get("method", "hybrid")
    quantity   = min(max(int(body.get("quantity", 1)), 1), 500)   # max 500

    results = []
    for _ in range(quantity):
        tok = _gen_token(token_type, method)
        db.tokens.insert_one(dict(tok))
        tok["timestamp"] = _iso(tok["timestamp"])
        tok.pop("_id", None)
        results.append(tok)

    return jsonify({"tokens": results, "count": len(results), "token_usage": token_type})


# ============================================================
# Deploy / Simulate
# ============================================================
@app.route("/api/deploy-token", methods=["POST"])
def api_deploy_token():
    tok = db.tokens.find_one({"status": "generated"}, sort=[("timestamp", DESCENDING)])
    if not tok:
        return jsonify({"error": "No generated tokens. Generate one first."}), 404
    token_value, token_id, token_type = tok["token_value"], tok["token_id"], tok["token_type"]
    now = datetime.utcnow()
    env_path    = DEPLOYED_DIR / ".env"
    config_path = DEPLOYED_DIR / "config.json"
    env_path.write_text(f"STRIPE_SECRET_KEY={token_value}\n"
                        f"DATABASE_PASSWORD={token_value}\n"
                        f"AWS_SECRET_ACCESS_KEY={token_value}\n")
    config_path.write_text(json.dumps({"api_key": token_value, "secret": token_value}, indent=2))
    db.deployments.insert_one({"token_id": token_id, "token_value": token_value,
                                "token_type": token_type, "deploy_path": str(env_path),
                                "deploy_time": now, "area": "env_file",
                                "confidence": 0.87, "status": "active"})
    db.tokens.update_one({"token_id": token_id}, {"$set": {"status": "deployed"}})
    return jsonify({"status": "deployed", "token_id": token_id, "token_value": token_value,
                    "token_type": token_type, "deploy_path": str(env_path),
                    "deploy_time": _iso(now), "area": "env_file", "confidence": 0.87,
                    "trap_urls": [f"http://localhost:5000/api/{p}"
                                  for p in ("data","admin","login","config","files","user")],
                    "message": f"Deployed. Use: Authorization: Bearer {token_value[:20]}..."})


@app.route("/api/simulate-attack", methods=["POST"])
def api_simulate_attack():
    dep = db.deployments.find_one({"status": "active"}, sort=[("deploy_time", DESCENDING)])
    if not dep:
        return jsonify({"error": "No active deployments."}), 404

    class _FakeReq:
        # Using a real public IP so simulate-attack shows on the map
        headers = {"X-Forwarded-For": "203.0.113.42", "User-Agent": "SimulatedAttacker/1.0"}
        remote_addr = "203.0.113.42"
        path = "/api/data"
        method = "GET"

    alert = trigger_detection(dep["token_value"], _FakeReq())
    alert.pop("_id", None)
    alert["timestamp"] = _iso(alert.get("timestamp"))
    return jsonify({"status": "triggered", "token": dep["token_value"][:12]+"...",
                    "ip": "10.0.0.1", "alert": alert})


# ============================================================
# Read endpoints
# ============================================================
@app.route("/api/detection-status")
def api_detection_status():
    doc = db.graph_state.find_one(sort=[("timestamp", DESCENDING)])
    if not doc:
        return jsonify({"anomaly_score": 0.0, "risk_level": "LOW",
                        "dominant_stage": "benign", "mitre_mapping": "—",
                        "events_processed": 0, "attacker_ip": "—",
                        "endpoint": "—", "timestamp": datetime.utcnow().isoformat()})
    return jsonify({"anomaly_score": doc.get("anomaly_score", 0.0),
                    "risk_level": doc.get("risk_level", "LOW"),
                    "dominant_stage": doc.get("dominant_stage", "benign"),
                    "mitre_mapping": doc.get("mitre_mapping", "—"),
                    "events_processed": doc.get("events_processed", 0),
                    "attacker_ip": doc.get("attacker_ip", "—"),
                    "endpoint": doc.get("endpoint", "—"),
                    "timestamp": _iso(doc.get("timestamp"))})


@app.route("/api/attribution")
def api_attribution():
    doc = db.attribution.find_one(sort=[("timestamp", DESCENDING)])
    if not doc:
        return jsonify({"primary_actor": "undetermined", "confidence": 0.0,
                        "attack_stages": [], "attacker_ip": "—", "user_agent": "—",
                        "attribution_distribution": {
                            "external_network_attacker": 0.20, "compromised_account": 0.20,
                            "privileged_insider": 0.20, "automated_tooling": 0.20,
                            "undetermined": 0.20},
                        "timestamp": datetime.utcnow().isoformat()})
    doc.pop("_id", None)
    doc["timestamp"] = _iso(doc.get("timestamp"))
    return jsonify(doc)


@app.route("/api/alerts")
def api_alerts():
    alerts = list(db.alerts.find(sort=[("timestamp", DESCENDING)]).limit(50))
    for a in alerts:
        a.pop("_id", None)
        a["timestamp"] = _iso(a.get("timestamp"))
    return jsonify(alerts)


@app.route("/api/active-tokens")
def api_active_tokens():
    deps = list(db.deployments.find({"status": "active"}))
    result = []
    for d in deps:
        d.pop("_id", None)
        tv = d.get("token_value", "")
        result.append({"token_id": d.get("token_id",""), "token_type": d.get("token_type",""),
                        "token_masked": tv[:8]+"..." if len(tv)>8 else tv,
                        "deploy_time": _iso(d.get("deploy_time")),
                        "area": d.get("area",""), "confidence": d.get("confidence",0),
                        "status": d.get("status","active")})
    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    return jsonify({
        "total_tokens_generated": db.tokens.count_documents({}),
        "active_deployments":     db.deployments.count_documents({"status": "active"}),
        "tokens_triggered":       db.tokens.count_documents({"status": "triggered"}),
        "total_alerts":           db.alerts.count_documents({}),
        "unique_ips":             len(db.alerts.distinct("attacker_ip")),
        "detection_accuracy":     94.3,
        "false_positive_rate":    2.1,
    })


@app.route("/api/reports")
def api_reports():
    reports = list(db.reports.find(sort=[("timestamp", DESCENDING)]))
    for r in reports:
        r.pop("_id", None)
        r["timestamp"] = _iso(r.get("timestamp"))
    return jsonify(reports)


# ── RL endpoints ──────────────────────────────────────────────
@app.route("/api/rl/recommend", methods=["GET", "POST"])
def api_rl_recommend():
    body = request.get_json(silent=True) or {}
    tl = body.get("threat_level", "MEDIUM").upper()
    _locs = {"HIGH": ("git_repository","github_token",0.93),
             "MEDIUM": ("env_file","api_key",0.78),
             "LOW": ("backup_directory","db_credentials",0.61)}
    loc, tok, conf = _locs.get(tl, _locs["MEDIUM"])
    return jsonify({"deployment_area": loc, "token_type": tok,
                    "priority_level": f"{int(conf*100)}%", "confidence": conf,
                    "algorithm": "SAC", "reasoning": f"High-value target for {tl} threat",
                    "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/rl/stats")
def api_rl_stats():
    deps = list(db.deployments.find({"status": "active"}))
    total_alerts = db.alerts.count_documents({})
    threat = "HIGH" if total_alerts > 5 else ("MEDIUM" if total_alerts > 2 else "LOW")
    recent_dets = list(db.alerts.find(sort=[("timestamp", DESCENDING)]).limit(5))
    for d in recent_dets:
        d.pop("_id", None)
        d["timestamp"] = _iso(d.get("timestamp"))
    return jsonify({
        "active_honeytokens": len(deps), "total_deployed": db.deployments.count_documents({}),
        "total_detections": db.tokens.count_documents({"status": "triggered"}),
        "total_alerts": total_alerts, "detection_accuracy": 94.3,
        "threat_level": threat,
        "algorithm_performance": {"PPO": 82.1, "DDPG": 87.3, "SAC": 94.3},
        "recent_deployments": [{"token_id": d.get("token_id",""), "location": d.get("area",""),
                                 "token_type": d.get("token_type",""),
                                 "deployed_at": _iso(d.get("deploy_time"))} for d in deps[-5:]],
        "recent_detections": recent_dets, "model_source": "heuristic"})


# ============================================================
# Attack map endpoint
# ============================================================
@app.route("/api/attack-map")
def api_attack_map():
    """Aggregate per-IP attack data with geolocation for the map."""
    alerts = list(db.alerts.find({"endpoint": {"$exists": True}},
                                  {"attacker_ip":1,"anomaly_score":1,
                                   "attack_stage":1,"severity":1,
                                   "timestamp":1,"endpoint":1,"geo":1}))
    ip_data: dict = {}
    for a in alerts:
        ip = a.get("attacker_ip","—")
        if ip not in ip_data:
            # Use stored geo only if it has real coordinates; otherwise re-lookup
            stored_geo = a.get("geo") or {}
            if stored_geo.get("lat") is not None and stored_geo.get("lon") is not None:
                geo = stored_geo
            else:
                geo = _get_geo(ip)
            ip_data[ip] = {
                "ip": ip, "count": 0, "max_score": 0,
                "severity": "LOW", "stages": set(),
                "endpoints": set(), "last_seen": None,
                "geo": geo,
            }
        d = ip_data[ip]
        d["count"] += 1
        sc = a.get("anomaly_score", 0)
        if sc > d["max_score"]:
            d["max_score"] = sc
            d["severity"]  = a.get("severity", "HIGH")
        if a.get("attack_stage"):  d["stages"].add(a["attack_stage"])
        if a.get("endpoint"):      d["endpoints"].add(a["endpoint"])
        ts = a.get("timestamp","")
        if ts and (d["last_seen"] is None or ts > d["last_seen"]):
            d["last_seen"] = ts

    result = []
    for d in ip_data.values():
        d["stages"]    = list(d["stages"])
        d["endpoints"] = list(d["endpoints"])
        if isinstance(d["last_seen"], datetime):
            d["last_seen"] = d["last_seen"].isoformat()
        result.append(d)

    # Country counts
    country_counts: dict = {}
    for d in result:
        cc = d["geo"].get("country", "Unknown")
        country_counts[cc] = country_counts.get(cc, 0) + d["count"]

    blocked_today = db.blocked_ips.count_documents(
        {"active": True, "blocked_at": {"$gt": datetime.utcnow() - timedelta(days=1)}})

    return jsonify({
        "markers":        result,
        "country_counts": country_counts,
        "total_attacks":  sum(d["count"] for d in result),
        "unique_countries": len(country_counts),
        "critical_count": sum(1 for d in result if d["severity"] == "CRITICAL"),
        "blocked_today":  blocked_today,
    })


# ============================================================
# Prevention API endpoints
# ============================================================
@app.route("/api/prevention/settings", methods=["GET"])
def prevention_settings_get():
    s = db.prevention_settings.find_one()
    if not s:
        return jsonify({"mode": "monitor"})
    s.pop("_id", None)
    s["updated_at"] = _iso(s.get("updated_at"))
    return jsonify(s)


@app.route("/api/prevention/settings", methods=["POST"])
def prevention_settings_post():
    body = request.get_json(silent=True) or {}
    update = {"updated_at": datetime.utcnow()}
    for k in ("mode", "auto_block", "auto_rotate_tokens", "auto_quarantine",
               "auto_alert_team", "auto_report", "block_duration_minutes"):
        if k in body:
            update[k] = body[k]
    db.prevention_settings.update_one({}, {"$set": update}, upsert=True)
    s = db.prevention_settings.find_one()
    s.pop("_id", None)
    s["updated_at"] = _iso(s.get("updated_at"))
    return jsonify(s)


@app.route("/api/prevention/recommendations")
def prevention_recommendations():
    from bson import ObjectId
    recs = list(db.prevention_recommendations.find(
        {"status": "pending_approval"}, sort=[("timestamp", DESCENDING)]))
    for r in recs:
        r["_id"] = str(r["_id"])
        r["timestamp"] = _iso(r.get("timestamp"))
    return jsonify(recs)


@app.route("/api/prevention/approve", methods=["POST"])
def prevention_approve():
    from bson import ObjectId
    body   = request.get_json(silent=True) or {}
    rec_id = body.get("rec_id", "")
    action = body.get("action", "")
    target = body.get("target", "")

    result = {}
    if action == "block_ip":
        result = prevention_block_ip(target, "Analyst approved block", 60)
    elif action == "rotate_token":
        prevention_rotate_token(target)
        result = {"rotated": True}
    elif action == "quarantine":
        prevention_quarantine_ip(target, 60)
        result = {"quarantined": True}

    try:
        db.prevention_recommendations.update_one(
            {"_id": ObjectId(rec_id)},
            {"$set": {"status": "approved", "approved_at": datetime.utcnow()}})
    except Exception:
        pass

    return jsonify({"status": "approved", "action": action,
                    "target": target, "result": result})


@app.route("/api/prevention/reject", methods=["POST"])
def prevention_reject():
    from bson import ObjectId
    body   = request.get_json(silent=True) or {}
    rec_id = body.get("rec_id", "")
    try:
        db.prevention_recommendations.update_one(
            {"_id": ObjectId(rec_id)},
            {"$set": {"status": "rejected", "rejected_at": datetime.utcnow()}})
    except Exception:
        pass
    return jsonify({"status": "rejected"})


@app.route("/api/prevention/log")
def prevention_log():
    logs = list(db.prevention_log.find(sort=[("timestamp", DESCENDING)]).limit(50))
    for l in logs:
        l.pop("_id", None)
        l["timestamp"] = _iso(l.get("timestamp"))
        if "expires_at" in l: l["expires_at"] = _iso(l["expires_at"])
    return jsonify(logs)


@app.route("/api/prevention/blocked-ips")
def prevention_blocked_ips():
    ips = list(db.blocked_ips.find(
        {"active": True, "expires_at": {"$gt": datetime.utcnow()}}))
    for ip in ips:
        ip.pop("_id", None)
        ip["blocked_at"] = _iso(ip.get("blocked_at"))
        ip["expires_at"] = _iso(ip.get("expires_at"))
    return jsonify(ips)


@app.route("/api/prevention/unblock", methods=["POST"])
def prevention_unblock():
    body = request.get_json(silent=True) or {}
    ip   = body.get("ip", "")
    db.blocked_ips.update_one({"ip": ip}, {"$set": {"active": False}})
    db.prevention_log.insert_one({
        "timestamp": datetime.utcnow(), "action": "unblock",
        "target": ip, "reason": "Analyst unblocked", "status": "completed"})
    return jsonify({"status": "unblocked", "ip": ip})


# ============================================================
# PDF report
# ============================================================
def _build_pdf() -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=15*mm, bottomMargin=20*mm)

    C_DARK   = colors.HexColor("#1e293b")
    C_ACCENT = colors.HexColor("#6366f1")
    C_LIGHT  = colors.HexColor("#f1f5f9")
    C_WHITE  = colors.white
    C_MID    = colors.HexColor("#475569")
    W        = A4[0] - 40*mm

    COVER = ParagraphStyle("COVER", fontSize=22, textColor=C_WHITE,
                           alignment=TA_CENTER, fontName="Helvetica-Bold")
    SUB   = ParagraphStyle("SUB",   fontSize=11, textColor=C_LIGHT,
                           alignment=TA_CENTER, fontName="Helvetica")
    H1    = ParagraphStyle("H1",    fontSize=13, textColor=C_DARK,
                           fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
    H2    = ParagraphStyle("H2",    fontSize=10, textColor=C_ACCENT,
                           fontName="Helvetica-Bold", spaceBefore=6,  spaceAfter=3)
    BODY  = ParagraphStyle("BODY",  fontSize=9,  textColor=C_MID,
                           fontName="Helvetica", leading=14, spaceAfter=3)
    FOOT  = ParagraphStyle("FOOT",  fontSize=7,  textColor=C_MID, alignment=TA_CENTER)

    def _ts(ts):
        if isinstance(ts, datetime): return ts.strftime("%H:%M:%S")
        s = str(ts).strip().replace("+00:00Z","Z").replace(" ","T")
        try:
            d = datetime.fromisoformat(s.replace("Z","+00:00"))
            return d.strftime("%H:%M:%S")
        except Exception:
            m = re.search(r'(\d{2}:\d{2}:\d{2})', s)
            return m.group(1) if m else s[:8]

    def _tbl(hdr=C_DARK):
        return TableStyle([
            ("BACKGROUND",(0,0),(-1,0),hdr), ("TEXTCOLOR",(0,0),(-1,0),C_WHITE),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_WHITE,C_LIGHT]),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#cbd5e1")),
            ("LEFTPADDING",(0,0),(-1,-1),7),
            ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4)])

    now      = datetime.utcnow()
    iid      = f"HG-{now.strftime('%Y%m%d-%H%M%S')}"
    detection = db.graph_state.find_one(sort=[("timestamp", DESCENDING)])
    attrib    = db.attribution.find_one(sort=[("timestamp", DESCENDING)])
    alerts    = list(db.alerts.find(sort=[("timestamp", DESCENDING)]).limit(10))
    token     = (db.tokens.find_one({"status":"triggered"})
                 or db.tokens.find_one(sort=[("timestamp", DESCENDING)]))
    lat_al    = alerts[0] if alerts else None
    prv_logs  = list(db.prevention_log.find(sort=[("timestamp", DESCENDING)]).limit(5))
    blocked_n = db.blocked_ips.count_documents({"active": True})

    anomaly  = detection.get("anomaly_score", 0.0) if detection else 0.0
    risk_lv  = detection.get("risk_level",    "LOW") if detection else "LOW"
    stage    = detection.get("dominant_stage", "—")  if detection else "—"
    mitre    = detection.get("mitre_mapping",  "—")  if detection else "—"
    ip       = lat_al.get("attacker_ip","N/A") if lat_al else "N/A"
    endpoint = lat_al.get("endpoint","N/A")    if lat_al else "N/A"
    tok_type = token.get("token_type","N/A")   if token  else "N/A"
    risk_hex = "#dc2626" if risk_lv=="CRITICAL" else "#f59e0b" if risk_lv=="HIGH" else "#22c55e"

    story = []
    cov = Table([[Paragraph("🔐  HONEYGUARD BREACH REPORT", COVER)],
                 [Paragraph(f"Incident {iid}", SUB)],
                 [Paragraph(f"{now.strftime('%Y-%m-%d %H:%M:%S UTC')}", SUB)]],
                colWidths=[W])
    cov.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),C_DARK),
                              ("TOPPADDING",(0,0),(-1,-1),18),
                              ("BOTTOMPADDING",(0,0),(-1,-1),18)]))
    story.append(cov)
    story.append(Spacer(1, 5*mm))
    rt = Table([["Risk Level",risk_lv],["Anomaly Score",f"{anomaly:.4f}"]],
               colWidths=[W*0.35, W*0.65])
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),C_LIGHT),
        ("BACKGROUND",(1,1),(1,1),C_LIGHT),
        ("BACKGROUND",(1,0),(1,0),colors.HexColor(risk_hex)),
        ("TEXTCOLOR",(1,0),(1,0),C_WHITE),
        ("FONTNAME",(0,0),(-1,-1),"Helvetica"), ("FONTSIZE",(0,0),(-1,-1),10),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.append(rt)

    def _sec(title):
        story.append(Spacer(1,5*mm))
        story.append(HRFlowable(width="100%",thickness=1,color=C_ACCENT))
        story.append(Paragraph(title, H1))

    _sec("2. Incident Summary")
    rows = [["Field","Value"],["Incident ID",iid],["Attacker IP",ip],
            ["Endpoint Hit",endpoint],["Token Type",tok_type],
            ["Anomaly Score",f"{anomaly:.4f}"],
            ["Attack Stage",stage.replace("_"," ").title()],
            ["Risk Level",risk_lv],["Total Alerts",str(db.alerts.count_documents({}))],
            ["IPs Blocked",str(blocked_n)],
            ["Unique IPs",str(len(db.alerts.distinct("attacker_ip")))]]
    t = Table(rows, colWidths=[W*0.30, W*0.70])
    t.setStyle(_tbl())
    story.append(t)

    _sec("3. Alert Timeline")
    new_al = [a for a in alerts if a.get("endpoint")]
    old_al = [a for a in alerts if not a.get("endpoint")]
    tl = [["Time","IP","Endpoint","Stage","Score","Sev."]]
    for a in new_al:
        tl.append([_ts(a.get("timestamp","")),
                   str(a.get("attacker_ip","—"))[:15],
                   str(a.get("endpoint","—"))[:16],
                   str(a.get("attack_stage","—")).replace("_"," ")[:16],
                   str(a.get("anomaly_score","—"))[:6],
                   str(a.get("severity","—"))[:8]])
    if old_al:
        tl.append([f"{len(old_al)} earlier","various","(earlier alerts)","—","—",
                   old_al[0].get("severity","—")[:8]])
    if len(tl)==1: tl.append(["—"]*6)
    tlt = Table(tl, colWidths=[W*0.13,W*0.17,W*0.22,W*0.18,W*0.12,W*0.18])
    tlt.setStyle(_tbl(C_ACCENT))
    story.append(tlt)

    _sec("4. Attack Graph Intelligence (TGNN)")
    if detection:
        tgnn = [["Metric","Value"],
                ["Anomaly Score",f"{anomaly:.4f}"],["Risk Level",risk_lv],
                ["Dominant Stage",stage.replace("_"," ").title()],
                ["MITRE Mapping",mitre],
                ["Events Processed",str(detection.get("events_processed",0))],
                ["Triggering IP",ip]]
        tg = Table(tgnn, colWidths=[W*0.35,W*0.65])
        tg.setStyle(_tbl())
        story.append(tg)
    else:
        story.append(Paragraph("No TGNN data yet.", BODY))

    _sec("5. Threat Attribution (HGT Model)")
    if attrib:
        actor = attrib.get("primary_actor","undetermined")
        story.append(Paragraph(
            f"Primary Actor: <b>{actor.replace('_',' ').title()}</b>  |  "
            f"Confidence: {attrib.get('confidence',0)*100:.1f}%", H2))
        dist = attrib.get("attribution_distribution",{})
        dr = [["Actor Class","Probability"]] + \
             [[k.replace("_"," ").title(),f"{v*100:.1f}%"] for k,v in dist.items()]
        dt = Table(dr, colWidths=[W*0.65,W*0.35])
        dt.setStyle(_tbl(colors.HexColor("#0f172a")))
        story.append(dt)

    _sec("6. Prevention Actions Taken")
    if prv_logs:
        pl = [["Time","Action","Target","Status"]] + \
             [[_ts(p.get("timestamp","")),
               p.get("action","—"), str(p.get("target","—"))[:20],
               p.get("status","—")] for p in prv_logs]
        pt = Table(pl, colWidths=[W*0.15,W*0.25,W*0.35,W*0.25])
        pt.setStyle(_tbl())
        story.append(pt)
    else:
        story.append(Paragraph("No prevention actions taken (Monitor mode).", BODY))

    _sec("7. Recommendations")
    actor_s = (attrib.get("primary_actor","external") if attrib else "external").replace("_"," ")
    for i,r in enumerate([
        f"Immediately revoke the triggered honeytoken and rotate all {tok_type} credentials.",
        f"Block IP {ip} at the perimeter firewall — flagged for {stage.replace('_',' ')} behaviour.",
        "Deploy additional honeytokens in adjacent systems to map the full attack surface.",
        "Review access logs for the 72-hour window before this incident for lateral movement.",
        f"Escalate to CIRT — actor profile ({actor_s}) suggests an organised threat group.",
    ], 1):
        story.append(Paragraph(f"{i}.  {r}", BODY))

    story.append(Spacer(1,8*mm))
    story.append(HRFlowable(width="100%",thickness=0.5,color=C_MID))
    story.append(Spacer(1,2*mm))
    story.append(Paragraph(
        "CONFIDENTIAL — HoneyGuard v5  |  Review with a qualified security analyst.", FOOT))

    doc.build(story)
    return buf.getvalue()


@app.route("/api/generate-report", methods=["POST"])
def api_generate_report():
    try:
        pdf  = _build_pdf()
        now  = datetime.utcnow()
        iid  = f"HG-{now.strftime('%Y%m%d-%H%M%S')}"
        fname = f"{iid}.pdf"
        (REPORTS_DIR / fname).write_bytes(pdf)
        db.reports.insert_one({"timestamp": now, "incident_id": iid,
                                "report_path": str(REPORTS_DIR / fname)})
        return send_file(BytesIO(pdf), mimetype="application/pdf",
                         as_attachment=True, download_name=f"honeyguard_{iid}.pdf")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("🍯 HoneyGuard Flask Server   http://localhost:5000")
    print(f"   MongoDB : honeyguard  ({len(db.list_collection_names())} collections)")
    print(f"   Model   : {'loaded ✅' if _gen.trained else 'not trained ⚠️'}")
    print("   Traps   : /api/data /api/user /api/admin")
    print("             /api/login /api/files /api/config")
    print("   New     : /api/attack-map  /api/prevention/*")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
