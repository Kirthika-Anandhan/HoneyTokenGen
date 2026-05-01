# ============================================================
# alerts_router.py — Real-time SSE alert system (Module 5)
# Persists every alert to MongoDB `alerts` collection.
# ============================================================

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from database import db

def _now() -> datetime:
    return datetime.now(timezone.utc)

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])

# ── Shortcut to the collection ───────────────────────────────
_col = db["alerts"] if db is not None else None


# ============================================================
# Broadcaster — SSE fan-out + MongoDB persistence
# ============================================================
class AlertBroadcaster:
    def __init__(self):
        self._clients: list[asyncio.Queue] = []

    # ── SSE subscription ─────────────────────────────────────
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._clients.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._clients.remove(q)
        except ValueError:
            pass

    # ── Broadcast + persist ───────────────────────────────────
    async def broadcast(self, alert: dict) -> None:
        # ① Persist to MongoDB
        if _col is not None:
            try:
                doc = dict(alert)
                doc["created_at"] = _now()
                _col.insert_one(doc)
            except Exception as e:
                print(f"[Alerts] MongoDB write error: {e}")

        # ② Fan out to every connected SSE client
        dead: list[asyncio.Queue] = []
        for q in list(self._clients):
            try:
                q.put_nowait(alert)
            except asyncio.QueueFull:
                dead.append(q)
            except Exception:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)

    # ── History (reads from MongoDB) ──────────────────────────
    def history(self, limit: int = 100) -> list:
        if _col is None:
            return []
        try:
            docs = list(
                _col.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
            )
            return docs
        except Exception as e:
            print(f"[Alerts] MongoDB read error: {e}")
            return []

    # ── Stats (reads from MongoDB) ────────────────────────────
    def stats(self) -> dict:
        if _col is None:
            return {"total": 0, "by_severity": {}, "connected_clients": len(self._clients)}
        try:
            total  = _col.count_documents({})
            by_sev = {
                sev: _col.count_documents({"severity": sev})
                for sev in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
            }
            return {
                "total":             total,
                "by_severity":       by_sev,
                "connected_clients": len(self._clients),
            }
        except Exception as e:
            print(f"[Alerts] MongoDB stats error: {e}")
            return {"total": 0, "by_severity": {}, "connected_clients": len(self._clients)}


# Single shared instance — imported by adaptive_rl.py
broadcaster = AlertBroadcaster()


# ============================================================
# Helper: normalised alert dict
# ============================================================
def make_alert(
    token_id: str,
    token_type: str,
    attacker_ip: str,
    location: str,
    severity: str = "HIGH",
    message: str = "Honeytoken accessed",
    extra: Optional[dict] = None,
) -> dict:
    alert = {
        "id":          str(uuid.uuid4()),
        "token_id":    token_id,
        "token_type":  token_type,
        "attacker_ip": attacker_ip,
        "location":    location,
        "severity":    severity.upper(),
        "message":     message,
        "timestamp":   _now().isoformat() + "Z",
    }
    if extra:
        alert.update(extra)
    return alert


# ============================================================
# SSE generator
# ============================================================
async def _event_stream(q: asyncio.Queue, request: Request):
    yield "event: connected\ndata: {\"status\": \"connected\"}\n\n"
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                alert   = await asyncio.wait_for(q.get(), timeout=25)
                payload = json.dumps(alert)
                yield f"event: alert\ndata: {payload}\n\n"
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        pass


# ============================================================
# Routes
# ============================================================

@router.get("/stream")
async def sse_stream(request: Request):
    """SSE endpoint — connect with:  new EventSource('/api/alerts/stream')"""
    q = broadcaster.subscribe()
    return StreamingResponse(
        _event_stream(q, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history")
async def alert_history(limit: int = 100):
    """Last `limit` alerts from MongoDB (most recent first)."""
    return JSONResponse({"alerts": broadcaster.history(limit)})


@router.get("/stats")
async def alert_stats():
    """Counts by severity + connected SSE client count — reads from MongoDB."""
    return JSONResponse(broadcaster.stats())


@router.post("/trigger")
async def manual_trigger(body: dict):
    """Manually fire a test alert and persist it to MongoDB."""
    alert = make_alert(
        token_id=   body.get("token_id",    "manual"),
        token_type= body.get("token_type",  "unknown"),
        attacker_ip=body.get("attacker_ip", "0.0.0.0"),
        location=   body.get("location",    "unknown"),
        severity=   body.get("severity",    "HIGH"),
        message=    body.get("message",     "Honeytoken accessed — manual trigger"),
    )
    await broadcaster.broadcast(alert)
    return JSONResponse({"status": "broadcast", "alert": alert})
