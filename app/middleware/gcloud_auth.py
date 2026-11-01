# app/middleware/gcloud_auth.py
import os
import logging
from fastapi import Request, HTTPException
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

logger = logging.getLogger(__name__)

# Map service-account email -> app_id.
# This is the ONLY place that decides who is who. Adding a fifth app = one line.
ALLOWED_CALLERS = {
    "lulit-payer@betcare.iam.gserviceaccount.com": "odaflux",
    "shaleqa-payer@betcare.iam.gserviceaccount.com": "shaleqa",
    "ethiogram-payer@ethiogram-504013.iam.gserviceaccount.com": "ethiogram",
    "behalf-payer@agency-control-plane-506714.iam.gserviceaccount.com": "behalf",
}


def _audience() -> str:
    # Must exactly match the Cloud Run service URL; verified against the token's `aud`.
    return os.getenv("MY_PORTAL_URL", "https://my-portal-1045664496839.us-central1.run.app").rstrip("/")


async def require_gcloud_caller(request: Request) -> str:
    """Verify the Bearer ID token and return the caller's app_id.

    Raises 401 if no token, 401 if token invalid, 403 if token valid but
    the service account is not in ALLOWED_CALLERS.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing ID token")

    token = auth[7:].strip()
    try:
        claims = id_token.verify_token(
            token,
            google_requests.Request(),
            audience=_audience(),
        )
    except Exception as e:
        logger.warning("ID token rejected: %s", e)
        raise HTTPException(401, "Invalid ID token")

    email = (claims.get("email") or "").lower()
    app_id = ALLOWED_CALLERS.get(email)
    if not app_id:
        logger.warning("ID token from unrecognised caller: %s", email)
        raise HTTPException(403, f"Caller {email} not permitted")
    return app_id
