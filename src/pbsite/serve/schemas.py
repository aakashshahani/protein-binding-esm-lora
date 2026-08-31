"""Request/response models for the inference API."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..data.clape import AA20


class PredictRequest(BaseModel):
    sequence: str = Field(..., min_length=1, max_length=4000, description="Amino-acid sequence")
    threshold: float | None = Field(None, ge=0.0, le=1.0, description="Override binding threshold")

    @field_validator("sequence")
    @classmethod
    def _validate_seq(cls, v: str) -> str:
        v = v.strip().upper()
        bad = set(v) - AA20
        if bad:
            raise ValueError(f"sequence contains non-standard residues: {sorted(bad)}")
        return v


class PredictResponse(BaseModel):
    sequence: str
    length: int
    threshold: float
    probabilities: list[float]
    binding_residues: list[int]  # 0-based indices predicted as binding
    model: str
