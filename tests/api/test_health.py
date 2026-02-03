"""
Tests for FastAPI health endpoint
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_ok():
    """Test that the health endpoint returns 200 with expected payload"""
    response = client.get("/health")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "model-autopsy"


def test_health_endpoint_method_not_allowed():
    """Test that POST to health endpoint is not allowed"""
    response = client.post("/health")
    assert response.status_code == 405


def test_imports_smoke():
    """Test that key modules can be imported without errors"""
    # API imports
    import api
    import api.main
    import api.routes.runs
    
    # Engine imports
    import engine
    
    # Verify app is properly configured
    from api.main import app
    assert app.title == "Model Autopsy"
    assert app.version == "0.1.0"
