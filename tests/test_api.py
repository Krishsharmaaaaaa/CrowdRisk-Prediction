"""
Unit and Integration tests for FastAPI endpoints.
Verifies health check, demo samples listing, error handling, and demo retrieval.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "crowd-risk-api"
    assert "version" in data


def test_demo_samples_manifest():
    response = client.get("/api/demo/samples")
    assert response.status_code == 200
    manifest = response.json()
    assert isinstance(manifest, list)
    assert len(manifest) >= 1
    sample = manifest[0]
    assert "sample_id" in sample
    assert "title" in sample
    assert "peak_risk" in sample


def test_demo_sample_retrieval_and_404():
    # Fetch valid demo if any exists
    samples_resp = client.get("/api/demo/samples")
    assert samples_resp.status_code == 200
    samples = samples_resp.json()
    if samples:
        sample_id = samples[0]["sample_id"]
        detail_resp = client.get(f"/api/demo/{sample_id}")
        assert detail_resp.status_code == 200
        data = detail_resp.json()
        assert "analysis_id" in data
        assert "timeline" in data
        assert "summary" in data

    # Fetch invalid demo
    invalid_resp = client.get("/api/demo/non_existent_sample_xyz")
    assert invalid_resp.status_code == 404
