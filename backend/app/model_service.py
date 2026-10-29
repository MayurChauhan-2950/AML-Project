"""Lazy-loading, caching model service.

Models are loaded on first use (never at import or startup) so the API starts
fast and /api/health works even without TensorFlow. Missing TensorFlow or
model files raise ``ModelUnavailableError`` -> HTTP 503.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import config
from .errors import ModelUnavailableError
from .gradcam import GradCam, overlay_heatmap

log = logging.getLogger("brainscan.models")


class ModelServiceProtocol(Protocol):
    """Interface used by the routers (tests provide a fake implementation)."""

    def predict(self, array: np.ndarray, model_name: str) -> list[float]: ...

    def heatmap(self, array: np.ndarray, model_name: str, class_index: int) -> bytes: ...

    def is_loaded(self) -> bool: ...


class ModelService:
    def __init__(self, model_paths: dict[str, Path] | None = None) -> None:
        self._paths = model_paths or config.MODEL_PATHS
        self._models: dict[str, Any] = {}
        self._cams: dict[str, GradCam] = {}
        self._lock = threading.RLock()

    def is_loaded(self) -> bool:
        return bool(self._models)

    def _load(self, name: str) -> Any:
        """Return the cached model, loading it on first use."""
        with self._lock:
            if name in self._models:
                return self._models[name]
            path = self._paths.get(name)
            if path is None:
                raise ModelUnavailableError(f"Unknown model '{name}'.")
            if not path.is_file():
                raise ModelUnavailableError(f"Model file for '{name}' was not found ({path.name}).")
            try:
                import tensorflow as tf  # lazy import
            except ImportError:
                raise ModelUnavailableError(
                    "TensorFlow is not installed, so the model cannot run. "
                    "Install it with: pip install tensorflow"
                ) from None
            try:
                try:
                    model = tf.keras.models.load_model(str(path), compile=False)
                except Exception:  # noqa: BLE001 - Lambda layer may need unsafe deserialisation
                    log.info("Retrying load of %s with safe_mode=False", name)
                    model = tf.keras.models.load_model(str(path), compile=False, safe_mode=False)
            except Exception as exc:  # noqa: BLE001
                log.exception("Failed to load model %s", name)
                raise ModelUnavailableError(
                    f"Model '{name}' could not be loaded ({type(exc).__name__}). "
                    "Check the TensorFlow/Keras version (Keras 3 / TensorFlow >= 2.16 is required)."
                ) from None
            self._models[name] = model
            log.info("Loaded model %s", name)
            return model

    def predict(self, array: np.ndarray, model_name: str) -> list[float]:
        """Class probabilities for one (224, 224, 3) raw-RGB [0, 255] array."""
        model = self._load(model_name)
        probs = model.predict(array[None, ...], verbose=0)[0]
        if len(probs) != len(config.CLASSES):
            raise ModelUnavailableError("Model output does not match the expected 4 classes.")
        return [float(p) for p in probs]

    def heatmap(self, array: np.ndarray, model_name: str, class_index: int) -> bytes:
        """Grad-CAM overlay PNG for ``class_index``."""
        model = self._load(model_name)
        with self._lock:
            cam = self._cams.get(model_name)
            if cam is None:
                cam = GradCam(model, config.GRADCAM_LAYERS.get(model_name))
                self._cams[model_name] = cam
        return overlay_heatmap(array, cam.heatmap(array[None, ...], class_index))
