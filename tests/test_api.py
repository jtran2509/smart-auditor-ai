import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()['status'] == "healthy"

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()

def test_invoice_upload_no_file():
    response = client.post("/api/v1/invoices/upload") 
    assert response.status_code == 422  # Missing file