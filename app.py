#!/usr/bin/env python3
"""PHI-Shield FastAPI service.

Endpoints:
  GET  /health       -> {"status": "ok"|"degraded", "backend": ...}
  POST /deidentify   -> {"text": ..., "mode": ...} in,
                        {"deidentified_text": ..., "entities": [...]} out
  GET  /             -> one-page demo UI (paste text, see before/after)

Run:  uvicorn app:app --host 127.0.0.1 --port 8000

Safety posture (security-audit 2026-10-31, findings D1/D2):
  * Fail-closed backend: the analyzer is built at startup; if no
    usable spaCy NER model is available the service refuses to start
    rather than silently degrading to pattern-only (which cannot see
    free-text names/places).  Pattern-only remains available only via
    the explicit PHI_SHIELD_ALLOW_PATTERN_ONLY=1 opt-in, and every
    response then carries a prominent "degraded" marker.
  * Bounded work per request: text is capped at MAX_TEXT_CHARS and
    the raw body at MAX_BODY_BYTES (413), and /deidentify is rate
    limited per client (429).
"""

import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from phi_shield.deidentify import MODES, Deidentifier

# --- Request bounds (finding D2) ---------------------------------------
# A long clinical note runs a few thousand characters; 100k chars
# (~15k words, dozens of pages) is generous headroom for any single
# note.  The cap also bounds worst-case CPU: the audit measured a
# 933,324-char submission taking ~98 s under the spaCy backend
# (~9.5k chars/s), so 100k chars keeps one request near ~10 s of
# pipeline time -- and the rate limit below bounds how often even
# that can repeat.
MAX_TEXT_CHARS = 100_000
# Raw JSON body ceiling, enforced before parsing.  A 100k-char text,
# JSON-escaped, fits comfortably under 1 MiB; anything larger is not
# a clinical note.
MAX_BODY_BYTES = 1_048_576
# Per-client fixed-window rate limit for POST /deidentify.
RATE_LIMIT_PER_MINUTE = int(
    os.environ.get("PHI_SHIELD_RATE_LIMIT_PER_MINUTE", "60")
)


class _RateLimiter:
    """In-process fixed-window limiter keyed by client host.

    The service runs as a single uvicorn process (see module
    docstring); behind a multi-worker deployment the same limit must
    also be enforced at the reverse proxy.
    """

    def __init__(self, limit_per_minute):
        self.limit = limit_per_minute
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key):
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - 60.0:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


_rate_limiter = _RateLimiter(RATE_LIMIT_PER_MINUTE)

_deidentifier = None


def get_deidentifier():
    global _deidentifier
    if _deidentifier is None:
        _deidentifier = Deidentifier()
    return _deidentifier


@asynccontextmanager
async def lifespan(app):
    # Build eagerly so a deployment without a usable NER model fails
    # at startup (fail-closed) instead of on the first request.
    get_deidentifier()
    yield


app = FastAPI(title="PHI-Shield", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            too_big = int(content_length) > MAX_BODY_BYTES
        except ValueError:
            too_big = False
        if too_big:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "request body too large",
                    "max_body_bytes": MAX_BODY_BYTES,
                },
            )
    return await call_next(request)


class DeidentifyRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mode: str = Field(default="label")


@app.get("/health")
def health():
    deid = get_deidentifier()
    body = {
        "status": "degraded" if deid.degraded else "ok",
        "backend": deid.backend,
    }
    if deid.degraded:
        body["degraded"] = True
    return body


@app.post("/deidentify")
def deidentify(req: DeidentifyRequest, request: Request):
    if req.mode not in MODES:
        return {"error": f"unknown mode {req.mode!r}; expected one of {MODES}"}
    if len(req.text) > MAX_TEXT_CHARS:
        return JSONResponse(
            status_code=413,
            content={
                "error": "text too large",
                "max_text_chars": MAX_TEXT_CHARS,
            },
        )
    client = request.client.host if request.client else "unknown"
    if not _rate_limiter.allow(client):
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate limit exceeded",
                "limit_per_minute": _rate_limiter.limit,
            },
            headers={"Retry-After": "60"},
        )
    deid = get_deidentifier()
    out, entities = deid.deidentify(req.text, mode=req.mode)
    body = {
        "deidentified_text": out,
        "mode": req.mode,
        "entities": entities,
        "backend": deid.backend,
    }
    if deid.degraded:
        body["degraded"] = True
        body["warning"] = (
            "DEGRADED pattern-only backend: no spaCy NER model is "
            "loaded, so free-text person/location names may NOT be "
            "redacted. Do not treat this output as fully de-identified."
        )
    return body


@app.get("/")
def demo():
    return FileResponse(Path(__file__).resolve().parent / "static" / "demo.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
