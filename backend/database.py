# ============================================================
# database.py — MongoDB integration for all 5 modules
# Database: honeyguard
# ============================================================

import copy
import os
from datetime import datetime

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME   = "honeyguard"

# ── Connection ───────────────────────────────────────────────
try:
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[DB_NAME]
    print(f"✅ MongoDB connected → {DB_NAME}")
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    print(f"⚠️  MongoDB connection failed: {e}")
    client = None
    db     = None

# ── Unified cross-module collections ────────────────────────
# Module 1 writes, Module 3 + Report read
tokens_col           = db["tokens"]           if db is not None else None
# Module 2 writes, Module 4 + Report read
graph_state_col      = db["graph_state"]      if db is not None else None
# Module 3 writes, Report reads
deployment_state_col = db["deployment_state"] if db is not None else None
# Module 4 writes, Report reads
attribution_col      = db["attribution"]      if db is not None else None
# Module 5 writes, Report reads
alerts_col           = db["alerts"]           if db is not None else None

# ── Legacy type-specific token collections (kept for DB partitioning) ──
if db is not None:
    COLLECTION_MAP = {
        "db_record": db["db_tokens"],
        "api":       db["api_tokens"],
        "jwt":       db["jwt_tokens"],
        "cloud":     db["cloud_tokens"],
        "github":    db["github_tokens"],
    }
else:
    COLLECTION_MAP = {}


# ============================================================
# Module 1 — Token storage
# ============================================================
def save_token(token_type: str, token_value: str, entropy: float,
               similarity: float, discriminator: float) -> dict:
    """
    Save a generated honeytoken.
    Writes to both the type-specific legacy collection AND the unified
    `tokens` collection so reports and the RL module can read it.
    """
    if not client:
        print("⚠️  MongoDB not connected — token not saved")
        return {"error": "Database not connected"}

    token_type = token_type.lower()
    if token_type not in COLLECTION_MAP:
        raise ValueError(f"Invalid token type: {token_type}. Must be one of: {list(COLLECTION_MAP)}")

    document = {
        "token_value":        token_value,
        "token_type":         token_type,
        "entropy":            entropy,
        "entropy_ratio":      similarity,
        "similarity":         similarity,
        "authenticity_score": similarity,
        "disc_score":         discriminator,
        "discriminator":      discriminator,
        "created_at":         datetime.utcnow(),
        "accessed":           False,
        "access_count":       0,
    }

    try:
        # Write to type-specific collection (legacy)
        doc_legacy = copy.deepcopy(document)
        result     = COLLECTION_MAP[token_type].insert_one(doc_legacy)

        # Write to unified tokens collection (new)
        doc_unified = copy.deepcopy(document)
        db["tokens"].insert_one(doc_unified)

        document["_id"] = str(result.inserted_id)
        print(f"✅ Token saved: {token_type} (ID: {result.inserted_id})")
        return document

    except Exception as e:
        print(f"❌ Error saving token: {e}")
        raise


# ============================================================
# Legacy helpers (unchanged — still work on type-specific cols)
# ============================================================
def get_token_by_id(token_type: str, token_id: str) -> dict:
    if not client:
        return {"error": "Database not connected"}
    token_type = token_type.lower()
    if token_type not in COLLECTION_MAP:
        raise ValueError(f"Invalid token type: {token_type}")
    from bson.objectid import ObjectId
    try:
        token = COLLECTION_MAP[token_type].find_one({"_id": ObjectId(token_id)})
        if token:
            token["_id"] = str(token["_id"])
        return token
    except Exception as e:
        print(f"❌ Error retrieving token: {e}")
        return None


def mark_token_accessed(token_type: str, token_id: str) -> bool:
    if not client:
        return False
    token_type = token_type.lower()
    if token_type not in COLLECTION_MAP:
        raise ValueError(f"Invalid token type: {token_type}")
    from bson.objectid import ObjectId
    try:
        result = COLLECTION_MAP[token_type].update_one(
            {"_id": ObjectId(token_id)},
            {
                "$set": {"accessed": True, "last_accessed": datetime.utcnow()},
                "$inc": {"access_count": 1},
            },
        )
        if result.modified_count > 0:
            print(f"⚠️  ALERT: Honeytoken accessed! Type: {token_type}, ID: {token_id}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error marking token as accessed: {e}")
        return False


def get_all_tokens(token_type: str = None, limit: int = 100) -> list:
    if not client:
        return []
    if token_type:
        token_type = token_type.lower()
        if token_type not in COLLECTION_MAP:
            raise ValueError(f"Invalid token type: {token_type}")
        tokens = list(COLLECTION_MAP[token_type].find().limit(limit))
    else:
        tokens = []
        for collection in COLLECTION_MAP.values():
            tokens.extend(list(collection.find().limit(limit // len(COLLECTION_MAP))))
    for token in tokens:
        token["_id"] = str(token["_id"])
    return tokens


def get_accessed_tokens() -> list:
    if not client:
        return []
    accessed = []
    for token_type, collection in COLLECTION_MAP.items():
        tokens = list(collection.find({"accessed": True}))
        for token in tokens:
            token["_id"]        = str(token["_id"])
            token["token_type"] = token_type
        accessed.extend(tokens)
    return accessed


def get_token_stats() -> dict:
    if not client:
        return {"error": "Database not connected"}
    stats = {
        "total_tokens": 0,
        "by_type": {},
        "accessed_count": 0,
        "average_entropy": 0.0,
        "average_authenticity": 0.0,
    }
    total_entropy = 0
    total_auth    = 0
    token_count   = 0
    for token_type, collection in COLLECTION_MAP.items():
        count    = collection.count_documents({})
        accessed = collection.count_documents({"accessed": True})
        stats["by_type"][token_type] = {"count": count, "accessed": accessed}
        stats["total_tokens"]  += count
        stats["accessed_count"] += accessed
        for token in collection.find({}, {"entropy": 1, "similarity": 1}):
            total_entropy += token.get("entropy", 0)
            total_auth    += token.get("similarity", 0)
            token_count   += 1
    if token_count > 0:
        stats["average_entropy"]      = round(total_entropy / token_count, 3)
        stats["average_authenticity"] = round(total_auth    / token_count, 3)
    return stats


def delete_token(token_type: str, token_id: str) -> bool:
    if not client:
        return False
    token_type = token_type.lower()
    if token_type not in COLLECTION_MAP:
        raise ValueError(f"Invalid token type: {token_type}")
    from bson.objectid import ObjectId
    try:
        result = COLLECTION_MAP[token_type].delete_one({"_id": ObjectId(token_id)})
        return result.deleted_count > 0
    except Exception as e:
        print(f"❌ Error deleting token: {e}")
        return False


def clear_all_tokens(token_type: str = None) -> int:
    if not client:
        return 0
    deleted_count = 0
    if token_type:
        token_type = token_type.lower()
        if token_type not in COLLECTION_MAP:
            raise ValueError(f"Invalid token type: {token_type}")
        result        = COLLECTION_MAP[token_type].delete_many({})
        deleted_count = result.deleted_count
    else:
        for collection in COLLECTION_MAP.values():
            result         = collection.delete_many({})
            deleted_count += result.deleted_count
    print(f"🗑️  Deleted {deleted_count} tokens")
    return deleted_count
