"""
Model Autopsy API
FastAPI service for ML model diagnostics
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Model Autopsy",
    description="Explain why machine learning models fail",
    version="0.1.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "model-autopsy"}


# Import routes after app is created to avoid circular imports
from api.routes import runs

app.include_router(runs.router, prefix="/runs", tags=["runs"])
