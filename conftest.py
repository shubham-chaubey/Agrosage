"""
Pytest shared fixtures for AgroSage.

Sets safe dummy environment variables BEFORE importing the app, so importing
`app` (which transitively imports the RAG engine) never needs a real API key.
The RAG engine's network calls are wrapped in try/except, so import is safe.
"""
import os

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
os.environ.setdefault("FLASK_SECRET_KEY", "ci-test-secret")

import pytest
import app as app_module


@pytest.fixture
def flask_app():
    app_module.app.config.update(TESTING=True)
    return app_module.app


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()
