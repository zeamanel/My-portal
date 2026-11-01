# app/services/ledger.py
"""Ledger state transitions.

The rules for moving a transaction out of `pending`, and for refunding one, live
here and nowhere else. The Chapa webhook, the dashboard's re-verify action, the
app-facing refund route and the dashboard's refund button all go through these
functions — a divergence between those paths would mean money recorded
differently depending on which one happened to run.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException

logger = logging.getLogger(__name__)

TABLE = "chapa_transactions"

# A transaction in one of these has been settled. Nothing re-settles it.
# Refunds are driven only by the refund functions, never by provider polling.
TERMINAL = {"success", "failed", "cancelled", "refunded"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def map_provider_status(
    provider_status: str,
    chapa_ref: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Map a Chapa status onto a ledger transition.

    Returns (new_status, update_fields). A new_status of None means "no
    transition" — the provider still considers it open, so the row is left
    pending rather than being guessed at.
    """
    status = (provider_status or "pending").strip().lower()

    if status == "success":
        update: Dict[str, Any] = {"status": "success", "succeeded_at": _now_iso()}
        if chapa_ref:
            update["chapa_ref"] = str(chapa_ref)
        return "success", update

    if status in ("failed", "failure", "error"):
        return "failed", {
            "status": "failed",
            "failed_at": _now_iso(),
            "failure_reason": f"provider reported {status}"[:500],
        }

    if status in ("cancelled", "canceled"):
        return "cancelled", {"status": "cancelled", "failed_at": _now_iso()}

    return None, {}


def extract_chapa_ref(verified: Dict[str, Any]) -> Optional[str]:
    data = (verified.get("raw") or {}).get("data") or {}
    if not isinstance(data, dict):
        return None
    ref = data.get("reference") or data.get("ref_id")
    return str(ref) if ref else None


def settle_from_verify(sb, tx_ref: str, verified: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a verified provider result to the ledger row.

    The caller is responsible for having checked that the row exists and is not
    already terminal.
    """
    provider_status = str(verified.get("status") or "pending")
    new_status, update = map_provider_status(
        provider_status, extract_chapa_ref(verified)
    )

    if new_status is None:
        logger.info(
            "%s: provider still reports %s; leaving pending", tx_ref, provider_status
        )
        return {
            "status": "pending",
            "tx_ref": tx_ref,
            "provider_status": provider_status,
            "changed": False,
        }

    sb.table(TABLE).update(update).eq("tx_ref", tx_ref).execute()
    logger.info("ledger %s -> %s", tx_ref, new_status)
    return {
        "status": new_status,
        "tx_ref": tx_ref,
        "provider_status": provider_status,
        "changed": True,
    }


async def reverify(sb, chapa, tx_ref: str, app_id: Optional[str] = None) -> Dict[str, Any]:
    """Re-ask Chapa about one transaction and settle it if it has moved.

    `app_id=None` means an admin call that may touch any business; a non-None
    value scopes the lookup so one app cannot reach another's transactions.
    """
    txn = _load(sb, tx_ref, app_id, "tx_ref, app_id, status")

    if txn["status"] in TERMINAL:
        return {
            "status": txn["status"],
            "tx_ref": tx_ref,
            "changed": False,
            "reason": "already settled",
        }

    try:
        verified = await chapa.verify(tx_ref)
    except Exception as e:
        logger.error("could not verify %s with Chapa: %s", tx_ref, e)
        raise HTTPException(502, f"Could not verify transaction with Chapa: {e}")

    return settle_from_verify(sb, tx_ref, verified)


async def refund_transaction(
    sb,
    chapa,
    tx_ref: str,
    amount_etb: Optional[float] = None,
    app_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Refund a successful payment.

    `amount_etb=None` refunds in full. `app_id=None` means an admin call that may
    touch any business; a non-None value scopes it to that app.
    """
    txn = _load(sb, tx_ref, app_id, "tx_ref, app_id, status, amount_etb, metadata")

    if txn["status"] != "success":
        raise HTTPException(
            409,
            f"Only a successful payment can be refunded (status: {txn['status']})",
        )

    try:
        raw = await chapa.refund(tx_ref, amount_etb)
    except Exception as e:
        logger.error("Chapa refund failed for %s: %s", tx_ref, e)
        raise HTTPException(502, f"Refund failed at provider: {e}")

    original = float(txn["amount_etb"])
    refunded = float(amount_etb) if amount_etb is not None else original

    # The schema has no refunded-amount column, so a partial refund would
    # otherwise be indistinguishable from a full one. Keep the amount in
    # metadata rather than losing it. See the note in the round-2 summary about
    # promoting this to a real column.
    metadata = dict(txn.get("metadata") or {})
    metadata["refund"] = {
        "amount_etb": refunded,
        "original_etb": original,
        "partial": refunded < original,
        "at": _now_iso(),
    }

    update = sb.table(TABLE).update(
        {"status": "refunded", "refunded_at": _now_iso(), "metadata": metadata}
    ).eq("tx_ref", tx_ref)
    if app_id is not None:
        update = update.eq("app_id", app_id)
    update.execute()

    logger.info("ledger %s -> refunded (%.2f of %.2f)", tx_ref, refunded, original)
    return {
        "tx_ref": tx_ref,
        "status": "refunded",
        "refunded_etb": refunded,
        "partial": refunded < original,
        "provider_response": raw,
    }


def _load(sb, tx_ref: str, app_id: Optional[str], columns: str) -> Dict[str, Any]:
    """Fetch one row, optionally scoped to an app.

    A tx_ref belonging to a different app raises 404 rather than 403, so the
    endpoint never confirms that another business's reference exists.
    """
    query = sb.table(TABLE).select(columns).eq("tx_ref", tx_ref)
    if app_id is not None:
        query = query.eq("app_id", app_id)
    res = query.execute()
    if not res.data:
        raise HTTPException(404, "Transaction not found")
    return res.data[0]
