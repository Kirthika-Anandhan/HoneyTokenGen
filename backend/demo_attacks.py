"""
demo_attacks.py — Simulate attacks from multiple real public IPs
Run: python3 demo_attacks.py

Generates + deploys a fresh token for each attacker so every
hit registers in the database and appears on the Attack Map.
"""
import requests, time

BASE = "http://127.0.0.1:5000"

ATTACKERS = [
    {"ip": "8.8.8.8",       "label": "Google DNS     (USA)",       "ua": "Mozilla/5.0 (Windows)"},
    {"ip": "1.1.1.1",       "label": "Cloudflare DNS (Australia)", "ua": "curl/7.88.1"},
    {"ip": "77.88.8.8",     "label": "Yandex DNS    (Russia)",     "ua": "Python-requests/2.28"},
    {"ip": "103.86.96.100", "label": "VPN node      (Singapore)",  "ua": "PostmanRuntime/7.32"},
    {"ip": "185.220.101.1", "label": "Tor Exit Node (Germany)",    "ua": "Wget/1.21"},
    {"ip": "203.0.113.42",  "label": "ISP network   (New York)",   "ua": "Go-http-client/1.1"},
]

print("=" * 58)
print("HoneyGuard — Multi-Country Attack Simulation")
print("=" * 58)

for atk in ATTACKERS:
    # Fresh token + deploy for every attacker
    gr = requests.post(f"{BASE}/api/generate-token",
                       json={"token_type": "jwt", "quantity": 1})
    dr = requests.post(f"{BASE}/api/deploy-token")
    if "error" in dr.json():
        print(f"  ✗ Deploy failed: {dr.json()}"); continue

    token = dr.json()["token_value"]

    ar = requests.get(f"{BASE}/api/data",
        headers={"Authorization": f"Bearer {token}",
                 "X-Forwarded-For": atk["ip"],
                 "User-Agent": atk["ua"]})

    print(f"  ✓ {atk['label']:<30} IP={atk['ip']}")
    time.sleep(1.2)   # ip-api.com rate limit

time.sleep(2)
print("\nChecking Attack Map...")
d = requests.get(f"{BASE}/api/attack-map").json()
real = [m for m in d["markers"]
        if m.get("geo",{}).get("lat") is not None
        and not m.get("geo",{}).get("private")]

print(f"\n{len(real)} markers on map:\n")
for m in sorted(real, key=lambda x: -x["count"]):
    g = m.get("geo", {})
    print(f"  {m['ip']:<22} → {g.get('city','?'):18} {g.get('country','?')}")

print("\n→ Refresh Attack Map tab to see all markers on the globe")
