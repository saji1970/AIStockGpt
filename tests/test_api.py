"""API endpoint smoke tests."""

import pytest


def test_health():
    """Test health endpoint returns 200."""
    from fastapi.testclient import TestClient
    from backend.main_enhanced import app

    client = TestClient(app)
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'healthy'


def test_status():
    """Test status endpoint returns 200."""
    from fastapi.testclient import TestClient
    from backend.main_enhanced import app

    client = TestClient(app)
    r = client.get('/status')
    assert r.status_code == 200
    assert r.json()['status'] == 'operational'


def test_root():
    """Test root endpoint returns 200."""
    from fastapi.testclient import TestClient
    from backend.main_enhanced import app

    client = TestClient(app)
    r = client.get('/')
    assert r.status_code == 200
    assert 'AI Stock GPT' in r.json()['message']
