"""Analyze + analyses CRUD endpoints (see docs/API_CONTRACT.md)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from .. import config
from ..deps import get_model_service, get_storage
from ..model_service import ModelServiceProtocol
from ..preprocess import sanitize_filename, to_model_array, validate_upload
from ..report import build_report_pdf
from ..storage import Storage

log = logging.getLogger("brainscan.api")
router = APIRouter(prefix="/api", tags=["analyses"])

# Uploaded medical images must not be cached by browsers/proxies.
_NO_STORE = {"Cache-Control": "private, no-store"}
_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


# -- serialisation ---------------------------------------------------------
def _summary(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": rec["id"],
        "created_at": rec["created_at"],
        "filename": rec["filename"],
        "predicted_class": rec["predicted_class"],
        "predicted_label": config.CLASS_LABELS.get(rec["predicted_class"], rec["predicted_class"]),
        "confidence": rec["confidence"],
    }


def _full(rec: dict[str, Any], storage: Storage) -> dict[str, Any]:
    base = f"/api/analyses/{rec['id']}"
    data = _summary(rec)
    data.update(
        probabilities=rec["probabilities"],
        model_version=rec["model_version"],
        image_url=f"{base}/image",
        heatmap_url=f"{base}/heatmap" if storage.heatmap_path(rec) else None,
        disclaimer=config.DISCLAIMER,
    )
    return data


def _require(storage: Storage, analysis_id: str) -> dict[str, Any]:
    rec = storage.get(analysis_id)
    if rec is None:
        raise HTTPException(404, "Analysis not found.")
    return rec


# -- analyze ---------------------------------------------------------------
@router.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    model: str = Query(config.DEFAULT_MODEL),
    heatmap: bool = Query(True),
    storage: Storage = Depends(get_storage),
    service: ModelServiceProtocol = Depends(get_model_service),
) -> dict[str, Any]:
    if model not in config.MODEL_PATHS:
        raise HTTPException(400, f"Unknown model '{model}'. Use one of: {', '.join(config.MODEL_PATHS)}.")

    # Read at most limit+1 bytes so a huge upload cannot exhaust memory.
    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    validated = validate_upload(file.filename, data)
    array = to_model_array(validated.image)

    probs = service.predict(array, model)  # may raise ModelUnavailableError -> 503
    best = int(np.argmax(probs))
    predicted = config.CLASSES[best]
    probabilities = {cls: round(float(p), 6) for cls, p in zip(config.CLASSES, probs)}

    heatmap_png: bytes | None = None
    if heatmap:
        try:
            heatmap_png = service.heatmap(array, model, best)
        except Exception:  # noqa: BLE001 - heatmap is optional; the prediction still stands
            log.exception("Grad-CAM failed for model %s", model)

    rec = storage.create(
        filename=sanitize_filename(file.filename),
        predicted_class=predicted,
        confidence=probabilities[predicted],
        probabilities=probabilities,
        model_version=config.MODEL_VERSIONS[model],
        image_bytes=data,
        image_ext=validated.extension,
        heatmap_png=heatmap_png,
    )
    log.info("analysis %s: %s (%.3f) via %s", rec["id"], predicted, rec["confidence"], model)
    return _full(rec, storage)


# -- list / get / delete ---------------------------------------------------
@router.get("/analyses")
def list_analyses(
    search: str | None = Query(None, max_length=100),
    label: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    storage: Storage = Depends(get_storage),
) -> dict[str, Any]:
    if label and label not in config.CLASSES:
        raise HTTPException(400, f"Unknown label '{label}'. Use one of: {', '.join(config.CLASSES)}.")
    items, total = storage.list_analyses(search=search, label=label, page=page, page_size=page_size)
    return {"items": [_summary(r) for r in items], "total": total, "page": page, "page_size": page_size}


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, storage: Storage = Depends(get_storage)) -> dict[str, Any]:
    return _full(_require(storage, analysis_id), storage)


@router.delete("/analyses/{analysis_id}")
def delete_analysis(analysis_id: str, storage: Storage = Depends(get_storage)) -> dict[str, Any]:
    if not storage.delete(analysis_id):
        raise HTTPException(404, "Analysis not found.")
    return {"deleted": True, "id": analysis_id}


# -- files -----------------------------------------------------------------
@router.get("/analyses/{analysis_id}/image")
def get_image(analysis_id: str, storage: Storage = Depends(get_storage)) -> FileResponse:
    path = storage.image_path(_require(storage, analysis_id))
    if not path.is_file():
        raise HTTPException(404, "Image file not found.")
    return FileResponse(path, media_type=_MEDIA_TYPES.get(Path(path).suffix.lower(), "image/png"),
                        headers=_NO_STORE)


@router.get("/analyses/{analysis_id}/heatmap")
def get_heatmap(analysis_id: str, storage: Storage = Depends(get_storage)) -> FileResponse:
    path = storage.heatmap_path(_require(storage, analysis_id))
    if path is None:
        raise HTTPException(404, "No heatmap available for this analysis.")
    return FileResponse(path, media_type="image/png", headers=_NO_STORE)


@router.get("/analyses/{analysis_id}/report")
def get_report(analysis_id: str, storage: Storage = Depends(get_storage)) -> Response:
    rec = _require(storage, analysis_id)
    image_path = storage.image_path(rec)
    if not image_path.is_file():
        raise HTTPException(404, "Image file not found.")
    pdf = build_report_pdf(rec, image_path, storage.heatmap_path(rec))
    headers = {**_NO_STORE, "Content-Disposition": f'inline; filename="brainscan-report-{rec["id"]}.pdf"'}
    return Response(content=pdf, media_type="application/pdf", headers=headers)
