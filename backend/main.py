"""
Stock Prediction API - Production Version
Minimal FastAPI application focused on ML predictions.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import API_TITLE, API_VERSION, CORS_ORIGINS
from routes import predict, stocks
import re

# Initialize FastAPI app
app = FastAPI(
    title="Stock Prediction API",
    version="4.0",
    description="AI-powered stock prediction API"
)

# Custom CORS middleware to handle Vercel wildcard domains
@app.middleware("http")
async def cors_middleware(request, call_next):
    origin = request.headers.get("origin")
    response = await call_next(request)
    
    # Allow specific origins or any *.vercel.app domain
    if origin:
        if origin in CORS_ORIGINS or (origin.endswith(".vercel.app") and origin.startswith("https://")):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response

# Configure CORS for preflight requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["https://signalist-stock-ai.vercel.app"],  # Add your production URL explicitly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(predict.router, tags=["prediction"])
app.include_router(stocks.router, tags=["stocks"])

# Simple health check
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Signalist API",
        "version": "4.0"
    }


@app.on_event("startup")
async def startup_event():
    """Startup event."""
    print("✅ Signalist API ready!")


if __name__ == "__main__":
    import uvicorn
    from core.config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
