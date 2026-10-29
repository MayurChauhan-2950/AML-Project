"""GET /api/stats."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ..deps import get_storage
from ..storage import Storage

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats")
def stats(storage: Storage = Depends(get_storage)) -> dict[str, Any]:
    return storage.stats()
