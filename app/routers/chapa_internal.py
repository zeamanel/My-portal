# app/routers/chapa_internal.py
"""Service-to-service payment routes.

Every route here depends on `require_gcloud_caller`, which returns the calling
app's `app_id`. That value comes from the verified ID token and is NEVER read
from the request body — otherwise any permitted caller could write rows, or read
transactions, belonging to another business.

Reads and refunds are scoped to the caller's own `app_id`. A tx_ref that belongs
to a different app returns 404, not 403, so the endpoint does not confirm that
another business's reference exists.
"""
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from app.deps import portal_base_url, require_chapa, require_supabase
from app.middleware.gcloud_auth import require_gcloud_caller
from app.services.ledger import TABLE, refund_transaction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/chapa", tags=["chapa-internal"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InitiateRequest(BaseModel):
    external_user_id: str = Field(min_length=1, max_length=200)
    amount_etb: float = Field(gt=0, le=1_000_000)
    email: EmailStr
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    return_url: str = Field(min_length=1)
    phone_number: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RefundRequest(BaseModel):
    tx_ref: str = Field(min_length=1, max_length=200)
    # None refunds the full amount.
    amount_etb: Optional[float] = Field(default=None, gt=0)


@router.post("/initiate")
async def initiate(
    body: InitiateRequest,
    app_id: str = Depends(require_gcloud_caller),
):
    """Open a Chapa checkout and record the attempt.

    The ledger row is written *before* Chapa is called, so a payment can never
    exist at Chapa without a local record of it. If Chapa then fails, the row is
    marked failed rather than deleted — a failed attempt is data.
    """
    sb = require_supabase()
    chapa = require_chapa()

    tx_ref = f"{app_id}-{uuid.uuid4().hex}"

    row = {
        "tx_ref": tx_ref,
        "app_id": app_id,                      # from the token, not the body
        "external_user_id": body.external_user_id,
        "amount_etb": round(body.amount_etb, 2),
        "status": "pending",
        "metadata": body.metadata,
        "return_url": body.return_url,
    }
    try:
        sb.table(TABLE).insert(row).execute()
    except Exception as e:
        logger.exception("ledger insert failed for %s", tx_ref)
        raise HTTPException(500, f"Could not record payment attempt: {e}")

    try:
        result = await chapa.initiate(
            tx_ref=tx_ref,
            amount_etb=body.amount_etb,
            email=str(body.email),
            first_name=body.first_name,
            last_name=body.last_name,
            phone_number=body.phone_number,
            callback_url=f"{portal_base_url()}/chapa/webhook",
            return_url=body.return_url,
            title=body.title or app_id.title(),
            description=body.description or f"Payment via {app_id}",
            metadata={**body.metadata, "app_id": app_id},
        )
    except Exception as e:
        logger.error("Chapa initiate failed for %s: %s", tx_ref, e)
        try:
            sb.table(TABLE).update(
                {
                    "status": "failed",
                    "failed_at": _now_iso(),
                    "failure_reason": f"initiate: {e}"[:500],
                }
            ).eq("tx_ref", tx_ref).execute()
        except Exception:
            logger.exception("could not mark %s failed", tx_ref)
        raise HTTPException(502, f"Payment provider error: {e}")

    checkout_url = result["checkout_url"]
    try:
        sb.table(TABLE).update({"chapa_checkout_url": checkout_url}).eq(
            "tx_ref", tx_ref
        ).execute()
    except Exception:
        # The checkout is live and the row exists; losing the cached URL is not
        # worth failing the call over.
        logger.exception("could not store checkout_url for %s", tx_ref)

    return {"tx_ref": tx_ref, "checkout_url": checkout_url, "status": "pending"}


@router.get("/transactions/{tx_ref}")
async def get_transaction(
    tx_ref: str,
    app_id: str = Depends(require_gcloud_caller),
):
    """Poll one transaction. Scoped to the caller's own app_id."""
    sb = require_supabase()
    res = (
        sb.table(TABLE)
        .select(
            "tx_ref, app_id, external_user_id, amount_etb, currency, status, "
            "chapa_ref, chapa_checkout_url, metadata, created_at, updated_at, "
            "succeeded_at, failed_at, refunded_at, failure_reason"
        )
        .eq("tx_ref", tx_ref)
        .eq("app_id", app_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(404, "Transaction not found")
    return {"transaction": res.data[0]}


@router.post("/refund")
async def refund(
    body: RefundRequest,
    app_id: str = Depends(require_gcloud_caller),
):
    """Refund a successful payment. Scoped to the caller's own app_id."""
    return await refund_transaction(
        require_supabase(),
        require_chapa(),
        body.tx_ref,
        amount_etb=body.amount_etb,
        app_id=app_id,
    )
