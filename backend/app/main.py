"""App factory. Run with ``uvicorn app.main:app --reload`` from the backend folder."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .errors import register_error_handlers
from .model_service import ModelService, ModelServiceProtocol
from .routers import analyses, health, stats
from .storage import Storage


def create_app(
    storage: Storage | None = None, model_service: ModelServiceProtocol | None = None
) -> FastAPI:
    """Build the app. Tests pass a temp-dir ``Storage`` and a fake model service."""
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="BrainScan AI API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.storage = storage or Storage()  # creates folders + tables
    app.state.model_service = model_service or ModelService()  # models load lazily
    register_error_handlers(app)
    for module in (health, analyses, stats):
        app.include_router(module.router)
    return app


app = create_app()
