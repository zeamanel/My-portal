# app/deps.py
"""Shared, lazily-built clients for the app.* routers.

main.py builds its own module-level Supabase client at import time. The routers
cannot import that without a circular import (main imports the routers), so they
get their own accessor here. Both are stateless service-role HTTP clients, so
having two is harmless.

Everything here fails *closed* with a 503 rather than raising at import time —
main.py deliberately starts even when Supabase is unconfigured (see the comment
at main.py "Do NOT raise - let the app start anyway so we can debug"), and a
missing payment credential must not take the whole portal down with it.
"""
import os
import logging
from functools import lru_cache
from typing import Optional

from fastapi import HTTPException
from supabase import create_client, Client

from app.services.chapa_service import ChapaService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_supabase() -> Optional[Client]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        logger.error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set; ledger unavailable")
        return None
    return create_client(url, key)


def require_supabase() -> Client:
    """The ledger client, or 503."""
    sb = get_supabase()
    if sb is None:
        raise HTTPException(503, "Ledger unavailable: Supabase is not configured")
    return sb


def require_chapa() -> ChapaService:
    """A Chapa client, or 503.

    Built per request on purpose. ChapaService.__init__ raises when
    CHAPA_SECRET_KEY is absent; at module scope that would stop the whole
    service (and /health with it) from starting.
    """
    try:
        return ChapaService()
    except ValueError as e:
        logger.error("Chapa unavailable: %s", e)
        raise HTTPException(503, str(e))


def portal_base_url() -> str:
    """Public base URL of this service, used to build the Chapa callback."""
    return os.getenv(
        "MY_PORTAL_URL",
        "https://my-portal-1045664496839.us-central1.run.app",
    ).rstrip("/")


# Business-day bucketing and all dashboard date maths use Ethiopian local time,
# not UTC, so "today's collections" means today where the businesses operate.
LEDGER_TIMEZONE = "Africa/Addis_Ababa"

# Canonical app_id order. Drives the dashboard's fixed colour assignment, so a
# business keeps its colour when others are filtered out.
APP_IDS = ("odaflux", "shaleqa", "ethiogram", "behalf")
