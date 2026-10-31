"""Regression tests for security-audit 2026-10-31 findings D1 and D2.

D1 (HIGH): a missing/mis-pinned spaCy model silently degraded the
service to pattern-only, returning free-text PERSON/LOCATION PHI
unredacted under HTTP 200.  The backend must now fail closed, a bad
pin must not suppress installed fallback models, and pattern-only is
reachable only via explicit opt-in and is marked degraded.

D2 (MEDIUM): POST /deidentify accepted unbounded text with no rate
limit.  Oversized submissions must now be rejected (413) and bursts
rate limited (429).
"""

import pytest
import spacy.util
from fastapi.testclient import TestClient

import app as app_module
from phi_shield import engine
from phi_shield.deidentify import Deidentifier

client = TestClient(app_module.app)

CONTROL = "Patient: John Smith\nMRN: 8473625"


def _no_models_installed(monkeypatch):
    monkeypatch.setattr(spacy.util, "is_package", lambda name: False)


def test_no_model_fails_closed(monkeypatch):
    """Audit D1 reproduction, after fix: no model -> raise, never a
    pattern-only engine handed out silently."""
    _no_models_installed(monkeypatch)
    monkeypatch.delenv("PHI_SHIELD_ALLOW_PATTERN_ONLY", raising=False)
    with pytest.raises(RuntimeError):
        engine.build_analyzer()


def test_pattern_only_only_via_explicit_opt_in_and_marked(monkeypatch):
    _no_models_installed(monkeypatch)
    monkeypatch.setenv("PHI_SHIELD_ALLOW_PATTERN_ONLY", "1")
    deid = Deidentifier()
    assert deid.degraded is True
    assert engine.backend_is_degraded(deid.backend)


def test_bad_pin_falls_back_to_installed_model(monkeypatch):
    """Audit D1 sub-case: one bad PHI_SHIELD_SPACY_MODEL pin used to
    suppress the installed fallback model entirely."""
    installed = [m for m in engine._MODEL_PREFERENCE if spacy.util.is_package(m)]
    if not installed:
        pytest.skip("no spaCy model installed in this environment")
    monkeypatch.setenv(
        "PHI_SHIELD_SPACY_MODEL", "definitely_not_installed_model_xyz"
    )
    _, backend = engine.build_analyzer()
    assert backend.startswith("spacy:")
    assert not engine.backend_is_degraded(backend)


def test_deidentify_rejects_oversized_text():
    big = "x" * (app_module.MAX_TEXT_CHARS + 1)
    res = client.post("/deidentify", json={"text": big, "mode": "label"})
    assert res.status_code == 413


def test_deidentify_rejects_oversized_body():
    # ~1.2 MB of UTF-8 JSON: rejected by the body-size middleware
    # before parsing.
    big = "é" * 600_000
    res = client.post("/deidentify", json={"text": big, "mode": "label"})
    assert res.status_code == 413


def test_deidentify_rate_limit(monkeypatch):
    monkeypatch.setattr(
        app_module, "_rate_limiter", app_module._RateLimiter(3)
    )
    payload = {"text": CONTROL, "mode": "label"}
    codes = [
        client.post("/deidentify", json=payload).status_code for _ in range(4)
    ]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


def test_deidentify_response_marks_degraded_backend(monkeypatch):
    class _Stub:
        backend = "pattern-only DEGRADED (test stub)"
        degraded = True

        def deidentify(self, text, mode="label"):
            return text, []

    monkeypatch.setattr(app_module, "get_deidentifier", lambda: _Stub())
    res = client.post("/deidentify", json={"text": "hello", "mode": "label"})
    assert res.status_code == 200
    body = res.json()
    assert body["degraded"] is True
    assert "warning" in body


def test_deidentify_response_not_degraded_on_spacy_backend():
    res = client.post("/deidentify", json={"text": CONTROL, "mode": "label"})
    assert res.status_code == 200
    body = res.json()
    assert "John Smith" not in body["deidentified_text"]
    assert not body.get("degraded", False)
