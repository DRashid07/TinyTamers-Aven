"""Tests for api/join.py: join requests from web/home.html (spam guards, checks, storage, the team's list).

Owner: C (Backend/LLM).
"""
import json

import pytest
from fastapi.testclient import TestClient

from api import join
from api.main import app

T0 = 1_800_000_000.0


@pytest.fixture
def client(monkeypatch, tmp_path):
    store = tmp_path / "requests.jsonl"
    monkeypatch.setenv("JOIN_REQUESTS_PATH", str(store))
    monkeypatch.delenv("JOIN_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("JOIN_FORM_SECRET", raising=False)
    clock = {"now": T0}
    monkeypatch.setattr(join, "_now", lambda: clock["now"])
    join.reset_limits()
    test_client = TestClient(app)
    test_client.clock = clock
    test_client.store = store
    return test_client


def body(client, wait=5, **overrides):
    """A valid request whose form was opened `wait` seconds ago."""
    token = client.get("/join-token").json()["token"]
    client.clock["now"] += wait
    data = {"name": "  Aysel   Məmmədova ", "phone": "050 123 45 67", "organization": "Bakı Dövlət Universiteti",
            "sector": "education", "purpose": "try", "consent": True, "website": "", "form_token": token}
    data.update(overrides)
    return data


def stored(client):
    if not client.store.exists():
        return []
    return [json.loads(line) for line in client.store.read_text(encoding="utf-8").splitlines()]


def test_valid_request_is_stored_with_a_clean_name_and_phone(client):
    response = client.post("/join-request", json=body(client))
    assert response.status_code == 200
    assert response.json() == {"ok": True, "message": join.SUCCESS}
    (record,) = stored(client)
    assert record["name"] == "Aysel Məmmədova"
    assert record["phone"] == "+994501234567"
    assert record["organization"] == "Bakı Dövlət Universiteti"
    assert (record["sector"], record["purpose"]) == ("education", "try")
    assert set(record) == {"id", "received_at", "name", "phone", "organization", "sector", "purpose"}


def test_token_is_not_cached(client):
    response = client.get("/join-token")
    assert response.headers["cache-control"] == "no-store"
    assert join.token_age(response.json()["token"]) == 0


@pytest.mark.parametrize("raw, expected", [
    ("+994 50 123 45 67", "+994501234567"),
    ("050-123-45-67", "+994501234567"),
    ("(012) 310 22 66", "+994123102266"),
    ("+994 10 555 12 34", "+994105551234"),
    ("+994 50 123 45 6", None),
    ("0012 345 67 89", None),
    ("+7 912 345 67 89", None),
    ("", None),
])
def test_normalize_phone(raw, expected):
    assert join.normalize_phone(raw) == expected


def test_form_sent_too_fast_is_refused_and_can_be_sent_again_later(client):
    data = body(client, wait=1)
    response = client.post("/join-request", json=data)
    assert response.status_code == 400 and response.json()["code"] == "too_fast"
    client.clock["now"] += 5
    assert client.post("/join-request", json=data).status_code == 200
    assert len(stored(client)) == 1


@pytest.mark.parametrize("token", ["", "abc", "123.deadbeef", "١٢٣.x", "9" * 150 + ".x"])
def test_missing_or_forged_token_is_refused(client, token):
    response = client.post("/join-request", json=body(client, form_token=token))
    assert response.status_code == 400 and response.json()["code"] == "token"
    assert stored(client) == []


def test_old_token_is_refused(client):
    response = client.post("/join-request", json=body(client, wait=join.TOKEN_MAX_AGE_S + 1))
    assert response.status_code == 400 and response.json()["code"] == "token"


def test_token_signed_with_another_secret_is_refused(client, monkeypatch):
    data = body(client)
    monkeypatch.setenv("JOIN_FORM_SECRET", "a-different-secret")
    assert client.post("/join-request", json=data).json()["code"] == "token"


def test_honeypot_gets_success_but_nothing_is_stored(client):
    response = client.post("/join-request", json=body(client, website="https://spam.example"))
    assert response.status_code == 200 and response.json()["ok"] is True
    assert stored(client) == []


@pytest.mark.parametrize("overrides, field", [
    ({"name": "Al"}, "name"),
    ({"name": "123 456"}, "name"),
    ({"name": "A" * 81}, "name"),
    ({"phone": "12345"}, "phone"),
    ({"organization": "Q" * 101}, "organization"),
    ({"sector": "restaurant"}, "sector"),
    ({"sector": ""}, "sector"),
    ({"purpose": "admin"}, "purpose"),
    ({"consent": False}, "consent"),
])
def test_invalid_field_is_named(client, overrides, field):
    response = client.post("/join-request", json=body(client, **overrides))
    assert response.status_code == 422
    reply = response.json()
    assert reply["ok"] is False and reply["field"] == field and reply["error"]
    assert stored(client) == []


def test_organization_is_optional(client):
    assert client.post("/join-request", json=body(client, organization="")).status_code == 200
    assert stored(client)[0]["organization"] == ""


def test_rate_limit_per_client_then_the_window_ends(client):
    for _ in range(join.CLIENT_LIMIT[0]):
        assert client.post("/join-request", json=body(client)).status_code == 200
    response = client.post("/join-request", json=body(client))
    assert response.status_code == 429
    assert 0 < int(response.headers["retry-after"]) <= join.CLIENT_LIMIT[1]
    client.clock["now"] += join.CLIENT_LIMIT[1]
    assert client.post("/join-request", json=body(client)).status_code == 200


def test_rate_limit_uses_the_address_added_by_the_proxy(client):
    for _ in range(join.CLIENT_LIMIT[0]):
        headers = {"X-Forwarded-For": "6.6.6.6, 10.0.0.1"}
        assert client.post("/join-request", json=body(client), headers=headers).status_code == 200
    # the browser-made first entry changes, the proxy-added last one does not: still the same client
    blocked = client.post("/join-request", json=body(client), headers={"X-Forwarded-For": "7.7.7.7, 10.0.0.1"})
    assert blocked.status_code == 429
    other = client.post("/join-request", json=body(client), headers={"X-Forwarded-For": "10.0.0.2"})
    assert other.status_code == 200


def test_global_limit_catches_changing_addresses(client):
    limit = join.GLOBAL_LIMIT[0]
    for i in range(limit):
        headers = {"X-Forwarded-For": f"10.1.{i // 250}.{i % 250}"}
        assert client.post("/join-request", json=body(client, wait=0, website="bot"), headers=headers).status_code == 200
    response = client.post("/join-request", json=body(client), headers={"X-Forwarded-For": "10.9.9.9"})
    assert response.status_code == 429


def test_full_store_refuses_new_requests(client, monkeypatch):
    monkeypatch.setattr(join, "MAX_STORE_BYTES", 10)
    assert client.post("/join-request", json=body(client)).status_code == 200
    response = client.post("/join-request", json=body(client))
    assert response.status_code == 503 and response.json()["ok"] is False
    assert len(stored(client)) == 1


def test_team_list_is_hidden_without_a_token(client):
    assert client.get("/join-requests").status_code == 404


def test_team_list_needs_the_right_token_and_shows_newest_first(client, monkeypatch):
    monkeypatch.setenv("JOIN_ADMIN_TOKEN", "team-secret")
    client.post("/join-request", json=body(client, name="Birinci Sorğu"))
    client.post("/join-request", json=body(client, name="İkinci Sorğu", purpose="support"))
    assert client.get("/join-requests").status_code == 401
    assert client.get("/join-requests", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/join-requests", headers={"Authorization": "Basic team-secret"}).status_code == 401
    response = client.get("/join-requests", headers={"Authorization": "Bearer team-secret"})
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    reply = response.json()
    assert reply["count"] == 2
    assert [r["name"] for r in reply["requests"]] == ["İkinci Sorğu", "Birinci Sorğu"]


def test_home_page_and_its_files_are_served(client):
    page = client.get("/home.html")
    assert page.status_code == 200 and "Qoşulma sorğusu" in page.text
    for asset in ("home.css", "home.js", "style.css", "ui.js"):
        assert client.get(f"/{asset}").status_code == 200
