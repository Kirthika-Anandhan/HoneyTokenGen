from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os
from datetime import datetime

# MongoDB configuration
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")

try:
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    # Test connection
    client.admin.command('ping')
    print("✅ MongoDB connected successfully")
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    print(f"⚠️  MongoDB connection failed: {e}")
    print("⚠️  Database operations will fail until MongoDB is available")
    client = None

# Database and collections
if client:
    db = client["honeytoken_db"]
    
    db_tokens_collection = db["db_tokens"]
    api_tokens_collection = db["api_tokens"]
    jwt_tokens_collection = db["jwt_tokens"]
    cloud_tokens_collection = db["cloud_tokens"]
    github_tokens_collection = db["github_tokens"]
    
    # Collection mapping
    COLLECTION_MAP = {
        "db_record": db_tokens_collection,
        "api": api_tokens_collection,
        "jwt": jwt_tokens_collection,
        "cloud": cloud_tokens_collection,
        "github": github_tokens_collection
    }
else:
    COLLECTION_MAP = {}


def save_token(token_type: str, token_value: str, entropy: float, 
               similarity: float, discriminator: float) -> dict:
    """
    Save generated honeytoken to MongoDB
    
    Args:
        token_type: Type of token (db_record, api, jwt, cloud, github)
        token_value: The token value (JSON string)
        entropy: Shannon entropy score
        similarity: Similarity/authenticity score
        discriminator: Discriminator score from ML model
    
    Returns:
        dict: Saved document with _id
    """
    if not client:
        print("⚠️  MongoDB not connected - token not saved")
        return {"error": "Database not connected"}
    
    token_type = token_type.lower()
    
    if token_type not in COLLECTION_MAP:
        raise ValueError(f"Invalid token type: {token_type}. Must be one of: {list(COLLECTION_MAP.keys())}")
    
    # Prepare document
    document = {
        "token_value": token_value,
        "entropy": entropy,
        "similarity": similarity,
        "authenticity_score": similarity,  # Alias for clarity
        "discriminator": discriminator,
        "token_type": token_type,
        "created_at": datetime.utcnow(),
        "accessed": False,
        "access_count": 0
    }
    
    try:
        # Insert into appropriate collection
        result = COLLECTION_MAP[token_type].insert_one(document)
        document['_id'] = str(result.inserted_id)
        print(f"✅ Token saved: {token_type} (ID: {result.inserted_id})")
        return document
    
    except Exception as e:
        print(f"❌ Error saving token: {e}")
        raise


def get_token_by_id(token_type: str, token_id: str) -> dict:
    """Retrieve a specific token by ID"""
    if not client:
        return {"error": "Database not connected"}
    
    token_type = token_type.lower()
    if token_type not in COLLECTION_MAP:
        raise ValueError(f"Invalid token type: {token_type}")
    
    from bson.objectid import ObjectId
    
    try:
        token = COLLECTION_MAP[token_type].find_one({"_id": ObjectId(token_id)})
        if token:
            token['_id'] = str(token['_id'])
        return token
    except Exception as e:
        print(f"❌ Error retrieving token: {e}")
        return None


def mark_token_accessed(token_type: str, token_id: str) -> bool:
    """
    Mark a honeytoken as accessed (indicates potential breach)
    """
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
                "$inc": {"access_count": 1}
            }
        )
        
        if result.modified_count > 0:
            print(f"⚠️  ALERT: Honeytoken accessed! Type: {token_type}, ID: {token_id}")
            return True
        return False
    
    except Exception as e:
        print(f"❌ Error marking token as accessed: {e}")
        return False


def get_all_tokens(token_type: str = None, limit: int = 100) -> list:
    """
    Retrieve all tokens of a specific type or all types
    """
    if not client:
        return []
    
    if token_type:
        token_type = token_type.lower()
        if token_type not in COLLECTION_MAP:
            raise ValueError(f"Invalid token type: {token_type}")
        
        tokens = list(COLLECTION_MAP[token_type].find().limit(limit))
    else:
        # Get from all collections
        tokens = []
        for collection in COLLECTION_MAP.values():
            tokens.extend(list(collection.find().limit(limit // len(COLLECTION_MAP))))
    
    # Convert ObjectId to string
    for token in tokens:
        token['_id'] = str(token['_id'])
    
    return tokens


def get_accessed_tokens() -> list:
    """
    Get all tokens that have been accessed (potential breaches)
    """
    if not client:
        return []
    
    accessed = []
    for token_type, collection in COLLECTION_MAP.items():
        tokens = list(collection.find({"accessed": True}))
        for token in tokens:
            token['_id'] = str(token['_id'])
            token['token_type'] = token_type
        accessed.extend(tokens)
    
    return accessed


def get_token_stats() -> dict:
    """
    Get statistics about generated honeytokens
    """
    if not client:
        return {"error": "Database not connected"}
    
    stats = {
        "total_tokens": 0,
        "by_type": {},
        "accessed_count": 0,
        "average_entropy": 0.0,
        "average_authenticity": 0.0
    }
    
    total_entropy = 0
    total_authenticity = 0
    token_count = 0
    
    for token_type, collection in COLLECTION_MAP.items():
        count = collection.count_documents({})
        accessed = collection.count_documents({"accessed": True})
        
        stats["by_type"][token_type] = {
            "count": count,
            "accessed": accessed
        }
        
        stats["total_tokens"] += count
        stats["accessed_count"] += accessed
        
        # Calculate averages
        tokens = list(collection.find({}, {"entropy": 1, "similarity": 1}))
        for token in tokens:
            total_entropy += token.get("entropy", 0)
            total_authenticity += token.get("similarity", 0)
            token_count += 1
    
    if token_count > 0:
        stats["average_entropy"] = round(total_entropy / token_count, 3)
        stats["average_authenticity"] = round(total_authenticity / token_count, 3)
    
    return stats


def delete_token(token_type: str, token_id: str) -> bool:
    """Delete a specific token"""
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
    """
    Clear all tokens (use with caution!)
    """
    if not client:
        return 0
    
    deleted_count = 0
    
    if token_type:
        token_type = token_type.lower()
        if token_type not in COLLECTION_MAP:
            raise ValueError(f"Invalid token type: {token_type}")
        result = COLLECTION_MAP[token_type].delete_many({})
        deleted_count = result.deleted_count
    else:
        # Clear all collections
        for collection in COLLECTION_MAP.values():
            result = collection.delete_many({})
            deleted_count += result.deleted_count
    
    print(f"🗑️  Deleted {deleted_count} tokens")
    return deleted_count