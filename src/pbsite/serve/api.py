"""FastAPI inference service: sequence -> per-residue binding probabilities.

The trained model is loaded lazily from ``$PBSITE_MODEL_DIR`` on first use. If
no model is present the /predict route returns 503 with a clear message, so the
app (and its health check / unit tests) run even before any training has happened.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from .. import __version__
from .schemas import PredictRequest, PredictResponse

app = FastAPI(title="pbsite — binding-site prediction", version=__version__)

_predictor = None
_load_error: str | None = None


def _get_predictor():
    global _predictor, _load_error
    if _predictor is not None:
        return _predictor
    model_dir = os.environ.get("PBSITE_MODEL_DIR", "")
    if not model_dir or not os.path.isdir(model_dir):
        _load_error = (
            "No trained model available. Set PBSITE_MODEL_DIR to a directory "
            "containing model.pt + meta.json (see scripts/train.py)."
        )
        return None
    try:
        from .predictor import Predictor

        _predictor = Predictor(model_dir, emb_cache=os.environ.get("PBSITE_EMB_DIR"))
    except Exception as exc:  # surface load errors instead of crashing the app
        _load_error = f"Failed to load model: {exc}"
        return None
    return _predictor


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "model_loaded": _get_predictor() is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail=_load_error)
    result = predictor.predict(req.sequence, threshold=req.threshold)
    return PredictResponse(
        sequence=req.sequence,
        length=len(req.sequence),
        probabilities=result["probabilities"],
        binding_residues=result["binding_residues"],
        threshold=result["threshold"],
        model=result["model"],
    )
