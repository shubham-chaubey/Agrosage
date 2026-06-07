"""
Flask route tests using the test client.

The Gemini-backed RAG call is monkeypatched so these tests are fast,
deterministic, and require no API key or network access.
"""
import app as app_module


def test_home_returns_200(client):
    assert client.get("/").status_code == 200


def test_about_returns_200(client):
    assert client.get("/about").status_code == 200


def test_features_returns_200(client):
    assert client.get("/features").status_code == 200


def test_query_page_returns_200(client):
    assert client.get("/query").status_code == 200


def test_recommendation_page_returns_200(client):
    assert client.get("/recommendation").status_code == 200


def test_unknown_route_returns_404(client):
    assert client.get("/this-route-does-not-exist").status_code == 404


def test_query_post_with_blank_question_returns_400(client):
    resp = client.post("/query", data={"question": "   "})
    assert resp.status_code == 400
    assert b"Please enter a question." in resp.data


def test_query_post_returns_mocked_answer(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_rag_answer", lambda q: "MOCK ANSWER")
    resp = client.post("/query", data={"question": "When should I sow wheat?"})
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["question"] == "When should I sow wheat?"
    assert payload["answer"] == "MOCK ANSWER"


def test_query_post_handles_engine_error_gracefully(client, monkeypatch):
    def boom(_q):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(app_module, "get_rag_answer", boom)
    resp = client.post("/query", data={"question": "anything"})
    # The route catches the exception and still returns 200 with an error message.
    assert resp.status_code == 200
    assert "Error" in resp.get_json()["answer"]


def test_profile_requires_login(client):
    resp = client.get("/profile")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]
