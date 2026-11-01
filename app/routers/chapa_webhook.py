# app/routers/chapa_webhook.py
"""Public Chapa webhook.

No auth dependency: Chapa cannot present a Google ID token. Authenticity rests
on the HMAC signature, and correctness rests on something stronger — the webhook
body is treated only as a *hint about which transaction to look at*. The ledger
is settled from an independent `verify` call to Chapa's API, via
`app.services.ledger`. A forged or malformed webhook therefore cannot move
money, even if the signature scheme below turns out to be wrong.

On the signature scheme: developer.chapa.co does not resolve from the build
environment, so the exact header and digest could not be confirmed against
Chapa's docs. Rather than guess one and risk rejecting every real payment, this
accepts any of the schemes Chapa is known to use. All of them require knowledge
of CHAPA_WEBHOOK_SECRET, so accepting several does not weaken the check. The
matched scheme is logged on every delivery — once a real payment arrives, read
the log, then pin `_candidates` to the single scheme observed.
"""
import base64
import hashlib
import hmac
import json
import logging
import os
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Request

from app.deps import require_chapa, require_supabase
from app.services.ledger import TABLE, TERMINAL, settle_from_verify

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chapa-webhook"])

SIGNATURE_HEADERS = ("Chapa-Signature", "x-chapa-signature")


def _candidates(raw: bytes, secret: str) -> Dict[str, str]:
    key = secret.encode()
    payload_mac = hmac.new(key, raw, hashlib.sha256)
    return {
        # HMAC over the raw body — the conventional scheme, and the only one
        # that binds the signature to the payload.
        "payload-hex": payload_mac.hexdigest(),
        "payload-b64": base64.b64encode(payload_mac.digest()).decode(),
        # HMAC of the secret with itself. Chapa has published examples using
        # this; it is a constant per secret, so it proves the sender knows the
        # secret but says nothing about the body. Tolerable here only because
        # the ledger is settled from verify(), not from the body.
        "secret-hex": hmac.new(key, key, hashlib.sha256).hexdigest(),
    }


def _verify_signature(raw: bytes, headers) -> str:
    """Return the name of the scheme that matched, or raise.

    Fails closed: with no secret configured, every delivery is rejected. A
    missing secret must never mean "skip verification".
    """
    secret = os.getenv("CHAPA_WEBHOOK_SECRET")
    if not secret:
        logger.error("CHAPA_WEBHOOK_SECRET not set; rejecting webhook")
        raise HTTPException(503, "Webhook verification is not configured")

    presented = []
    for name in SIGNATURE_HEADERS:
        value = headers.get(name)
        if value:
            presented.append((name, value.strip()))
    if not presented:
        raise HTTPException(401, "Missing signature header")

    candidates = _candidates(raw, secret)
    for header_name, presented_value in presented:
        for scheme, expected in candidates.items():
            # Both sides encoded to bytes: compare_digest rejects non-ASCII str
            # and the header is attacker-controlled.
            if hmac.compare_digest(
                presented_value.encode("utf-8", "replace"),
                expected.encode("ascii"),
            ):
                return f"{header_name}:{scheme}"

    logger.warning(
        "webhook signature mismatch (headers presented: %s)",
        [n for n, _ in presented],
    )
    raise HTTPException(401, "Invalid signature")


def _extract_tx_ref(payload) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if isinstance(data, dict) and data.get("tx_ref"):
        return str(data["tx_ref"])
    for key in ("tx_ref", "trx_ref", "reference"):
        if payload.get(key):
            return str(payload[key])
    return None


@router.post("/chapa/webhook")
async def chapa_webhook(request: Request):
    raw = await request.body()
    scheme = _verify_signature(raw, request.headers)

    try:
        payload = json.loads(raw or b"{}")
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Body is not valid JSON")

    tx_ref = _extract_tx_ref(payload)
    if not tx_ref:
        raise HTTPException(400, "No tx_ref in webhook payload")

    logger.info("chapa webhook for %s verified via %s", tx_ref, scheme)

    sb = require_supabase()
    res = (
        sb.table(TABLE)
        .select("tx_ref, app_id, status, amount_etb")
        .eq("tx_ref", tx_ref)
        .execute()
    )
    if not res.data:
        # 200 on purpose: a 4xx makes Chapa retry a reference we will never
        # recognise. Log it instead — it usually means a tx_ref from another
        # environment is pointed at this URL.
        logger.warning("webhook for unknown tx_ref %s", tx_ref)
        return {"status": "ignored", "reason": "unknown tx_ref"}

    txn = res.data[0]
    if txn["status"] in TERMINAL:
        return {
            "status": "already_settled",
            "tx_ref": tx_ref,
            "ledger_status": txn["status"],
        }

    # Source of truth. Never settle on what the webhook body claims.
    chapa = require_chapa()
    try:
        verified = await chapa.verify(tx_ref)
    except Exception as e:
        # 502 so Chapa retries — better to be told again than to settle blind.
        logger.error("could not verify %s with Chapa: %s", tx_ref, e)
        raise HTTPException(502, f"Could not verify transaction with Chapa: {e}")

    return settle_from_verify(sb, tx_ref, verified)
