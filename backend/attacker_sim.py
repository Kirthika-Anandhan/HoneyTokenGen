"""
attacker_sim.py — simulates an attacker who found a honeytoken and is using it.
The attacker sees only a normal API response. Detection fires silently on HoneyGuard.
"""
import requests
from pymongo import MongoClient

# Get the latest active deployed token from MongoDB
db = MongoClient()["honeyguard"]
dep = db.deployments.find_one({"status": "active"}, sort=[("deploy_time", -1)])

if not dep:
    print("❌ No active token deployed. Go to HoneyGuard → Deploy → Deploy Token first.")
    exit(1)

token = dep["token_value"]
BASE  = "http://127.0.0.1:5000"

print("=" * 55)
print("ATTACKER VIEW — using stolen token")
print(f"Token : {token[:30]}...")
print("=" * 55)

# Attacker tries different endpoints (like a real attacker would)
endpoints = [
    ("GET",  "/api/data"),
    ("GET",  "/api/user"),
    ("GET",  "/api/config"),
]

for method, path in endpoints:
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (attacker recon tool)",
    }
    resp = requests.request(method, BASE + path, headers=headers)
    print(f"\n{method} {path}")
    print(f"  Status   : {resp.status_code}")
    print(f"  Response : {resp.json()}")
    print(f"  [Attacker thinks: legitimate API response ✓]")

print("\n" + "=" * 55)
print("Meanwhile on HoneyGuard dashboard:")
print("  🔴 Breach panel slides in from right")
print("  📊 Anomaly score updates")
print("  📋 Alert logged to MongoDB")
print("=" * 55)
