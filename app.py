#!/usr/bin/env python3
"""PHI-Shield FastAPI service.

Endpoints:
  GET  /health       -> {"status": "ok", "backend": ...}
  POST /deidentify   -> {"text": ..., "mode": ...} in,
                        {"deidentified_text": ..., "entities": [...]} out
  GET  /             -> one-page demo UI (paste text, see before/after)

Run:  uvicorn app:app --host 127.0.0.1 --port 8000
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from phi_shield.deidentify import MODES, Deidentifier

app = FastAPI(title="PHI-Shield", version="0.1.0")
_deidentifier = None


def get_deidentifier():
    global _deidentifier
    if _deidentifier is None:
        _deidentifier = Deidentifier()
    return _deidentifier


class DeidentifyRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mode: str = Field(default="label")


@app.get("/health")
def health():
    return {"status": "ok", "backend": get_deidentifier().backend}


@app.post("/deidentify")
def deidentify(req: DeidentifyRequest):
    if req.mode not in MODES:
        return {"error": f"unknown mode {req.mode!r}; expected one of {MODES}"}
    deid = get_deidentifier()
    out, entities = deid.deidentify(req.text, mode=req.mode)
    return {"deidentified_text": out, "mode": req.mode, "entities": entities}


@app.get("/")
def demo():
    return FileResponse(Path(__file__).resolve().parent / "static" / "demo.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
