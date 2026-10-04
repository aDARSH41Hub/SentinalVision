from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query

from runtime_service import RuntimeConfig, RuntimeInferenceService, classify_probability

app = FastAPI(title="SentinelVision Runtime API", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    config = RuntimeConfig.from_project_root("D:/SentinalVision")
    return {
        "status": "ok",
        "project_root": str(config.project_root),
        "model_name": "TLF",
        "threshold": config.threshold,
        "audio_alpha": config.audio_alpha,
        "rgb_alpha": config.rgb_alpha,
    }


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    config = RuntimeConfig.from_project_root("D:/SentinalVision")
    return {
        "model_name": "TLF",
        "audio_model": str(config.audio_model_path),
        "rgb_model": str(config.rgb_model_path),
        "target_timesteps": config.target_timesteps,
        "audio_dim": config.audio_dim,
        "rgb_dim": config.rgb_dim,
        "threshold": config.threshold,
        "audio_alpha": config.audio_alpha,
        "rgb_alpha": config.rgb_alpha,
    }


@app.get("/predict")
def predict(
    segment_id: Optional[str] = Query(default=None, description="Frozen segment ID from the M3.5 manifest"),
    audio_probability: Optional[float] = Query(default=None),
    rgb_probability: Optional[float] = Query(default=None),
) -> Dict[str, Any]:
    if segment_id is not None:
        service = RuntimeInferenceService(project_root="D:/SentinalVision")
        result = service.predict_segment(segment_id)
        return result

    if audio_probability is None or rgb_probability is None:
        raise HTTPException(status_code=400, detail="Provide either segment_id or both audio_probability and rgb_probability.")

    fused_probability = 0.51 * float(audio_probability) + 0.49 * float(rgb_probability)
    label = classify_probability(fused_probability, threshold=0.49)
    return {
        "model_name": "TLF",
        "audio_probability": float(audio_probability),
        "rgb_probability": float(rgb_probability),
        "fused_probability": float(fused_probability),
        "decision_label": label,
        "decision_score": float(fused_probability),
        "threshold": 0.49,
        "alert_triggered": float(fused_probability) >= 0.49,
    }


@app.get("/status")
def status() -> Dict[str, Any]:
    return {
        "status": "initialized",
        "runtime": "SentinelVision Runtime API",
        "mode": "research-backed productization",
    }
