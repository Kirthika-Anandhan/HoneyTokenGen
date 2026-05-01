"""
test_all.py — Full HoneyGuard system test in one run
python3 test_all.py
"""
import requests, time, sys

BASE = "http://127.0.0.1:5000"
OK   = "✅"
FAIL = "❌"

def check(label, condition, detail=""):
    status = OK if condition else FAIL
    print(f"  {status} {label}" + (f"  ({detail})" if detail else ""))
    return condition

results = []

print("\n🍯 HoneyGuard — Full System Test")
print("=" * 50)

# ── 1. Health ─────────────────────────────────────
print("\n[1] Backend health")
r = requests.get(f"{BASE}/api/metrics")
results.append(check("Flask responding",        r.status_code == 200))
d = r.json()
results.append(check("MongoDB connected",       d["total_tokens_generated"] >= 0, f"{d['total_tokens_generated']} tokens"))
results.append(check("SSE endpoint reachable",  requests.get(f"{BASE}/api/events/stream",
                                                 stream=True, timeout=2).status_code == 200))

# ── 2. Token generation ───────────────────────────
print("\n[2] Token generation")
r = requests.post(f"{BASE}/api/generate-token",
                  json={"token_type":"jwt","quantity":3})
tokens = r.json().get("tokens", [])
results.append(check("Generate 3 tokens",       len(tokens) == 3, f"got {len(tokens)}"))
results.append(check("Auth score ≥ 0.90",       all(t["authenticity_score"] >= 0.90 for t in tokens),
                      f"min={min(t['authenticity_score'] for t in tokens):.3f}"))
results.append(check("Entropy ≥ 4.0",           all(t["entropy"] >= 4.0 for t in tokens)))

# ── 3. Deploy ─────────────────────────────────────
print("\n[3] Token deployment")
r = requests.post(f"{BASE}/api/deploy-token")
dep = r.json()
results.append(check("Deploy succeeds",         dep.get("status") == "deployed"))
results.append(check("Lure file written",       ".env" in dep.get("deploy_path",""),
                      dep.get("deploy_path","")))
token = dep.get("token_value","")

# ── 4. Trap endpoints ─────────────────────────────
print("\n[4] Trap endpoints (attacker gets fake 200)")
for path in ["/api/data","/api/user","/api/admin","/api/login","/api/files","/api/config"]:
    r = requests.get(f"{BASE}{path}")
    results.append(check(f"GET {path} → 200", r.status_code == 200))

# ── 5. Detection pipeline ─────────────────────────
print("\n[5] Detection pipeline")
before = requests.get(f"{BASE}/api/alerts").json()
r = requests.get(f"{BASE}/api/data",
    headers={"Authorization": f"Bearer {token}",
             "X-Forwarded-For": "8.8.8.8",
             "User-Agent": "TestRunner/1.0"})
results.append(check("Trap returns fake data",  r.json().get("status") == "success"))
time.sleep(1)
after = requests.get(f"{BASE}/api/alerts").json()
results.append(check("Alert recorded",          len(after) > len(before), f"+{len(after)-len(before)}"))
ds = requests.get(f"{BASE}/api/detection-status").json()
results.append(check("Anomaly score > 0",       ds["anomaly_score"] > 0, f"{ds['anomaly_score']}"))
results.append(check("Risk level set",          ds["risk_level"] in ("HIGH","CRITICAL"), ds["risk_level"]))
at = requests.get(f"{BASE}/api/attribution").json()
results.append(check("Attribution saved",       at.get("primary_actor") not in (None,""),
                      at.get("primary_actor")))

# ── 6. Attack map ─────────────────────────────────
print("\n[6] Attack map")
mp = requests.get(f"{BASE}/api/attack-map").json()
real = [m for m in mp["markers"] if m.get("geo",{}).get("lat") is not None
        and not m.get("geo",{}).get("private")]
results.append(check("Map endpoint responds",   "markers" in mp))
results.append(check("Public IP on map",        len(real) > 0, f"{len(real)} markers"))

# ── 7. Prevention ─────────────────────────────────
print("\n[7] Prevention system")
r = requests.get(f"{BASE}/api/prevention/settings")
results.append(check("Settings readable",       r.status_code == 200))
# Switch to recommend mode, trigger, check recs
requests.post(f"{BASE}/api/prevention/settings",
              json={"mode":"alert_recommend"})
requests.post(f"{BASE}/api/generate-token", json={"token_type":"api","quantity":1})
dep2 = requests.post(f"{BASE}/api/deploy-token").json()
if "token_value" in dep2:
    requests.get(f"{BASE}/api/data",
        headers={"Authorization": f"Bearer {dep2['token_value']}",
                 "X-Forwarded-For": "1.1.1.1"})
    time.sleep(1)
recs = requests.get(f"{BASE}/api/prevention/recommendations").json()
results.append(check("Recommendations generated", len(recs) > 0, f"{len(recs)} pending"))
requests.post(f"{BASE}/api/prevention/settings", json={"mode":"monitor"})

# ── 8. Blocked IP returns 429 ─────────────────────
print("\n[8] Block enforcement")
requests.post(f"{BASE}/api/prevention/approve",
              json={"rec_id": recs[0]["_id"] if recs else "x",
                    "action":"block_ip","target":"1.1.1.1"})
time.sleep(0.5)
r = requests.get(f"{BASE}/api/data",
    headers={"X-Forwarded-For":"1.1.1.1","Authorization":"Bearer anything"})
results.append(check("Blocked IP gets 429",     r.status_code == 429, f"got {r.status_code}"))
requests.post(f"{BASE}/api/prevention/unblock", json={"ip":"1.1.1.1"})

# ── 9. PDF report ─────────────────────────────────
print("\n[9] PDF report generation")
r = requests.post(f"{BASE}/api/generate-report")
results.append(check("PDF generated",           r.status_code == 200))
results.append(check("PDF valid",               r.headers.get("content-type","").startswith("application/pdf")))
results.append(check("PDF size > 3KB",          len(r.content) > 3000, f"{len(r.content)//1024}KB"))

# ── Summary ───────────────────────────────────────
passed = sum(results)
total  = len(results)
print(f"\n{'='*50}")
print(f"  Result: {passed}/{total} checks passed")
if passed == total:
    print(f"  {OK} All systems operational")
else:
    print(f"  {FAIL} {total-passed} check(s) failed")
print(f"{'='*50}\n")
sys.exit(0 if passed == total else 1)
