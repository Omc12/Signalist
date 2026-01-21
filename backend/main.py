"""
Stock Prediction API - Production Version
Minimal FastAPI application focused on ML predictions.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import API_TITLE, API_VERSION, CORS_ORIGINS
from routes import predict

# Initialize FastAPI app
app = FastAPI(
    title="Stock Prediction API",
    version="4.0",
    description="AI-powered stock prediction API"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main prediction router
app.include_router(predict.router, tags=["prediction"])

# Simple health check
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Stock Prediction API",
        "version": "4.0"
    }


@app.on_event("startup")
async def startup_event():
    """Startup event."""
    print("✅ Enhanced Stock Prediction API ready!")


if __name__ == "__main__":
    import uvicorn
    from core.config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
