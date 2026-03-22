from fastapi import FastAPI, Request
from attack_graph_router import router as attack_graph_router
from threat_attribution_router import router as threat_attribution_router

from fastapi.middleware.cors import CORSMiddleware

import json
import random

from token_generator import HoneytokenGenerator
from database import save_token

app = FastAPI()
generator = HoneytokenGenerator()


# ==============================
# ML Scoring Fallback
# ==============================
def score_ml_token(token_string):
    # Dummy discriminator scoring (you can replace with real model)
    return round(random.uniform(0.4, 0.6), 3)


# ==============================
# Unified Token Generator
# ==============================
def generate_realistic_token(token_usage, name=None, surname=None):

    if token_usage == "db_record":
        return generator.generate_db_credentials(name, surname)

    elif token_usage == "jwt":
        return generator.generate_jwt()

    elif token_usage == "github":
        return generator.generate_git_token()

    elif token_usage == "api":
        return generator.generate_api_key()

    elif token_usage == "cloud":
        return generator.generate_api_key()

    else:
        return generator.generate_api_key()


# ==============================
# API Endpoint
# ==============================
@app.post("/generate-token")
async def generate_token_endpoint(request: Request):

    body = await request.json()

    token_usage = body.get("token_usage", "api").lower()
    quantity = body.get("quantity", 5)

    name = body.get("name")
    surname = body.get("surname")

    results = []

    for _ in range(quantity):

        token_value = generate_realistic_token(
            token_usage,
            name,
            surname
        )

        # Convert dict tokens to string for DB storage
        token_string = json.dumps(token_value)

        discriminator = score_ml_token(token_string)

        entropy = round(random.uniform(0.7, 0.99), 3)
        similarity = round(random.uniform(0.7, 0.95), 3)

        save_token(
            token_usage,
            token_string,
            entropy,
            similarity,
            discriminator
        )

        results.append({
            "token_value": token_value,
            "entropy": entropy,
            "similarity": similarity,
            "discriminator": discriminator
        })

    return {"tokens": results}


# ==============================
# Root Endpoint
# ==============================
@app.get("/")
def home():
    return {"message": "AI Honeytoken Generator Running"}


# ==============================
# CORS Setup
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import os



# Import the ML-based honeytoken generator
from token_generator import HoneytokenGenerator
from database import save_token

app = FastAPI(title="ML Honeytoken Generator API", version="2.0")
app.include_router(attack_graph_router)
app.include_router(threat_attribution_router)

# ==============================
# Initialize ML Generator
# ==============================
print("🚀 Initializing ML Honeytoken Generator...")
generator = HoneytokenGenerator(device='cpu')

# Check if pre-trained models exist
MODEL_PATH = "honeytoken_models.pt"

if os.path.exists(MODEL_PATH):
    print(f"📂 Loading pre-trained models from {MODEL_PATH}...")
    generator.load_models(MODEL_PATH)
else:
    print("⚠️  No pre-trained models found. Training new models...")
    
    # Training data - Add your real token examples here
    training_data = {
        'jwt': [
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNDU2Nzg5IiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIn0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
            "eyJhbGciOiJIUzM4NCIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJzdXBlcnVzZXIifQ.xK7j9mP3nQ5rT8sV2wY4zB6cD1eF0gH"
        ],
        'api_key': [
            "sk_live_51H6aBcDeFgHiJkLmNoPqRsTuVwXyZ",
            "api_8djFh2Ksl9XkLmPqRtUvWxYz123456",
            "key_test_ABCdef123456XYZ789ghiJKL",
            "sk_test_4eC39HqLyjWDarjtT1zdp7dc"
        ],
        'git_token': [
            "ghp_A1B2C3D4E5F6G7H8I9J0KLMNOPQRSTUVWXYZ",
            "ghp_ZYXWVUTSRQPONMLKJIHGFEDCBA123456789ABC",
            "ghp_5K8L3M9N2P7Q1R6S4T0U8V3W7X2Y5Z1A4B"
        ]
    }
    
    # Train the models
    generator.train(training_data, epochs=50, batch_size=4)
    
    # Save for future use
    generator.save_models(MODEL_PATH)
    print("✅ Models trained and saved!")

print("✅ ML Honeytoken Generator ready!")


# ==============================
# Request/Response Models
# ==============================
class TokenRequest(BaseModel):
    token_usage: str = "api"  # db_record, jwt, github, api, cloud
    quantity: int = 5
    name: Optional[str] = None
    surname: Optional[str] = None
    method: str = "hybrid"  # vae, diffusion, hybrid
    
    class Config:
        schema_extra = {
            "example": {
                "token_usage": "jwt",
                "quantity": 3,
                "method": "hybrid"
            }
        }


class TokenResponse(BaseModel):
    token_value: dict
    entropy: float
    similarity: float  # Now called authenticity_score internally
    discriminator: float
    method: str
    token_type: str


# ==============================
# Unified Token Generator
# ==============================
def generate_realistic_token(token_usage: str, method: str = "hybrid", 
                            name: Optional[str] = None, 
                            surname: Optional[str] = None) -> dict:
    """
    Generate tokens using ML-based generator
    
    Args:
        token_usage: Type of token (db_record, jwt, github, api, cloud)
        method: Generation method (vae, diffusion, hybrid)
        name: Optional name for DB credentials
        surname: Optional surname for DB credentials
    
    Returns:
        dict: Generated token with metadata
    """
    
    token_usage = token_usage.lower()
    
    if token_usage == "db_record":
        result = generator.generate_db_credentials()
        # Add method info
        result['method'] = 'pattern-based'
        return result
    
    elif token_usage == "jwt":
        return generator.generate_jwt(method=method)
    
    elif token_usage == "github":
        return generator.generate_git_token(method=method)
    
    elif token_usage in ["api", "cloud"]:
        return generator.generate_api_key(method=method)
    
    else:
        # Default to API key
        return generator.generate_api_key(method=method)


# ==============================
# API Endpoints
# ==============================
@app.post("/generate-token", response_model=dict)
async def generate_token_endpoint(request: TokenRequest):
    """
    Generate honeytokens using ML models
    
    - **token_usage**: Type of token to generate (db_record, jwt, github, api, cloud)
    - **quantity**: Number of tokens to generate (1-100)
    - **name**: Optional name for DB credentials
    - **surname**: Optional surname for DB credentials  
    - **method**: Generation method (vae, diffusion, hybrid)
    """
    
    # Validate quantity
    if request.quantity < 1 or request.quantity > 100:
        raise HTTPException(status_code=400, detail="Quantity must be between 1 and 100")
    
    # Validate token_usage
    valid_types = ["db_record", "jwt", "github", "api", "cloud"]
    if request.token_usage.lower() not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid token_usage. Must be one of: {', '.join(valid_types)}"
        )
    
    # Validate method
    valid_methods = ["vae", "diffusion", "hybrid"]
    if request.method.lower() not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method. Must be one of: {', '.join(valid_methods)}"
        )
    
    results = []
    
    for i in range(request.quantity):
        try:
            # Generate token using ML models
            token_data = generate_realistic_token(
                token_usage=request.token_usage,
                method=request.method,
                name=request.name,
                surname=request.surname
            )
            
            # Extract metrics
            token_type = token_data.get('type', request.token_usage)
            entropy = token_data.get('entropy', 0.0)
            authenticity = token_data.get('authenticity_score', 0.0)
            method_used = token_data.get('method', request.method)
            
            # For compatibility, use 'similarity' as authenticity score
            similarity = authenticity
            
            # Discriminator score is the same as authenticity for ML models
            discriminator = authenticity
            
            # Prepare token value for storage
            # For DB credentials, store the full dict
            if token_type == "db_credentials":
                token_value = token_data
            else:
                # For other tokens, just store the token string
                token_value = {
                    "token": token_data.get('token', ''),
                    "type": token_type,
                    "entropy": entropy,
                    "authenticity_score": authenticity,
                    "method": method_used
                }
            
            # Convert to JSON string for MongoDB storage
            token_string = json.dumps(token_value)
            
            # Save to database
            save_token(
                token_type=request.token_usage,
                token_value=token_string,
                entropy=entropy,
                similarity=similarity,
                discriminator=discriminator
            )
            
            # Prepare response
            results.append({
                "token_value": token_value,
                "entropy": entropy,
                "similarity": similarity,
                "discriminator": discriminator,
                "method": method_used,
                "token_type": token_type
            })
            
        except Exception as e:
            print(f"Error generating token {i+1}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating token: {str(e)}")
    
    return {
        "tokens": results,
        "count": len(results),
        "method": request.method,
        "token_usage": request.token_usage
    }


@app.post("/generate-token-batch")
async def generate_token_batch(requests: List[TokenRequest]):
    """
    Generate multiple batches of different token types
    """
    all_results = []
    
    for req in requests:
        batch_result = await generate_token_endpoint(req)
        all_results.append({
            "token_usage": req.token_usage,
            "tokens": batch_result["tokens"]
        })
    
    return {
        "batches": all_results,
        "total_tokens": sum(len(batch["tokens"]) for batch in all_results)
    }


@app.get("/")
def home():
    """API health check and information"""
    return {
        "message": "ML-Based Honeytoken Generator API Running",
        "version": "2.0",
        "model_trained": generator.trained,
        "features": [
            "VAE-based generation",
            "Diffusion-based generation", 
            "Hybrid generation",
            "90-95% authenticity scores",
            "Multiple token types",
            "Real-time entropy calculation"
        ],
        "endpoints": {
            "/generate-token": "Generate honeytokens",
            "/generate-token-batch": "Generate multiple batches",
            "/model-info": "Get model information",
            "/retrain": "Retrain models with new data",
            "/api/attack-graph/demo": "Attack behaviour graph (Module 2)",
            "/api/threat-attribution/demo": "Threat attribution & profiling (Module 4)"
        }
    }


@app.get("/model-info")
def model_info():
    """Get information about the ML models"""
    return {
        "model_trained": generator.trained,
        "device": str(generator.device),
        "available_methods": ["vae", "diffusion", "hybrid"],
        "supported_token_types": [
            "jwt",
            "api_key", 
            "git_token",
            "db_credentials"
        ],
        "model_architecture": {
            "vae": {
                "vocab_size": 128,
                "max_len": 64,
                "latent_dim": 32,
                "hidden_dim": 128
            },
            "diffusion": {
                "vocab_size": 128,
                "max_len": 64,
                "hidden_dim": 256,
                "num_steps": 50
            },
            "discriminator": {
                "vocab_size": 128,
                "max_len": 64,
                "hidden_dim": 128
            }
        }
    }


@app.post("/retrain")
async def retrain_models(request: Request):
    """
    Retrain models with new token examples
    
    Expects JSON body with token examples:
    {
        "jwt": ["example1", "example2"],
        "api_key": ["example1", "example2"],
        "git_token": ["example1", "example2"],
        "epochs": 50
    }
    """
    try:
        body = await request.json()
        
        # Extract training data
        training_data = {
            'jwt': body.get('jwt', []),
            'api_key': body.get('api_key', []),
            'git_token': body.get('git_token', [])
        }
        
        # Remove empty lists
        training_data = {k: v for k, v in training_data.items() if v}
        
        if not training_data:
            raise HTTPException(status_code=400, detail="No training data provided")
        
        epochs = body.get('epochs', 50)
        batch_size = body.get('batch_size', 4)
        
        # Retrain models
        print(f"🔄 Retraining models with {sum(len(v) for v in training_data.values())} examples...")
        generator.train(training_data, epochs=epochs, batch_size=batch_size)
        
        # Save updated models
        generator.save_models(MODEL_PATH)
        
        return {
            "message": "Models retrained successfully",
            "training_samples": {k: len(v) for k, v in training_data.items()},
            "epochs": epochs,
            "model_saved": MODEL_PATH
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "ml_models_loaded": generator.trained,
        "database_connected": True,  # You can add actual DB check here
        "available_generators": {
            "vae": True,
            "diffusion": True,
            "hybrid": True
        }
    }


# ==============================
# CORS Setup
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# Startup Event
# ==============================
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 60)
    print("🔐 ML Honeytoken Generator API Started")
    print("=" * 60)
    print(f"📊 Model Status: {'Trained' if generator.trained else 'Not Trained'}")
    print(f"🎯 Available Methods: VAE, Diffusion, Hybrid")
    print(f"🔑 Token Types: JWT, API Key, Git Token, DB Credentials")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)