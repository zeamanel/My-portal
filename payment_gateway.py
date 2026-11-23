import os
import re
import hmac
import hashlib
import json
import uuid
import logging
import datetime
import secrets
from typing import Optional, Dict, Any, List, Iterable, Tuple
import httpx
from google.cloud import firestore
from google.api_core.exceptions import AlreadyExists
from gcp_client import get_firestore_client

logger = logging.getLogger(__name__)

# Secrets come from the environment only. The values that used to be hardcoded
# here as fallbacks are in git history and must be treated as compromised:
# rotate them in the Chapa dashboard and set the new ones on Cloud Run.
CHAPA_SECRET_KEY = os.getenv("CHAPA_SECRET_KEY", "")
CHAPA_PUBLIC_KEY = os.getenv("CHAPA_PUBLIC_KEY", "")
# "1234567890" was the old placeholder default; treat it as unset.
_webhook_secret_env = os.getenv("CHAPA_WEBHOOK_SECRET", "")
CHAPA_WEBHOOK_SECRET = "" if _webhook_secret_env == "1234567890" else _webhook_secret_env
CHAPA_API_URL = "https://api.chapa.co/v1"
MY_PORTAL_PUBLIC_URL = os.getenv("PUBLIC_URL", "https://my-portal-1045664496839.us-central1.run.app")

TX_REF_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
SIGNATURE_HEADERS = ("Chapa-Signature", "x-chapa-signature")


class GatewayError(Exception):
    """An error that maps to a specific HTTP status for the caller."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# Default registered apps. Credentials are deliberately NOT listed here: they
# are generated at seed time and live only in Firestore. Existing Firestore
# documents keep whatever keys they already have until rotated with
# rotate_app_credentials() / POST /api/admin/payments/apps/{app_id}/rotate.
DEFAULT_APPS = [
    {
        "app_id": "musa-ai",
        "name": "Musa AI / Odaflux",
        "webhook_url": "https://musa-ai-backend-1045664496839.us-central1.run.app/api/v1/chapa/webhook",
        "description": "Image/Video generation platform credits recharge",
        "is_active": True
    },
    {
        "app_id": "ethiogram",
        "name": "Ethiogram",
        "webhook_url": "https://ethiogram.com/api/billing/chapa-webhook",
        "description": "Ethiogram Telegram marketing platform balance",
        "is_active": True
    },
    {
        "app_id": "shaleqa",
        "name": "Shaleqa",
        "webhook_url": "https://shaleqa-backend-1045664496839.us-central1.run.app/api/v1/payments/chapa-callback",
        "description": "Shaleqa marketplace orders and customer payments",
        "is_active": True
    },
    {
        "app_id": "system-backend",
        "name": "System Backend",
        "webhook_url": "https://systembackendservice-ebjq5oaqna-uc.a.run.app/api/v1/payments/callback",
        "description": "Internal business workflows & vendor settlements",
        "is_active": True
    }
]


def _new_credentials(app_id: str) -> Dict[str, str]:
    slug = app_id[:12]
    return {
        "api_key": f"chp_live_{slug}_{secrets.token_hex(16)}",
        "webhook_secret": f"whsec_live_{slug}_{secrets.token_hex(16)}",
    }


def ensure_default_apps():
    """Seed the default client apps in Firestore if they are missing."""
    db = get_firestore_client()
    coll = db.collection("payment_apps")
    for app in DEFAULT_APPS:
        doc = coll.document(app["app_id"]).get()
        if not doc.exists:
            app_data = {**app, **_new_credentials(app["app_id"]), "created_at": _now()}
            coll.document(app["app_id"]).set(app_data)
            # Never log credentials; they are visible to admins on the dashboard.
            logger.info("Registered client app %s", app["app_id"])
        else:
            doc_data = doc.to_dict() or {}
            if not doc_data.get("webhook_secret"):
                coll.document(app["app_id"]).update({
                    "webhook_secret": _new_credentials(app["app_id"])["webhook_secret"]
                })


def rotate_app_credentials(app_id: str) -> Dict[str, Any]:
    """Issue a new API key and webhook secret for an app. The old ones stop working at once."""
    db = get_firestore_client()
    doc_ref = db.collection("payment_apps").document(app_id)
    if not doc_ref.get().exists:
        raise GatewayError(404, f"Unknown app_id: {app_id}")
    creds = _new_credentials(app_id)
    doc_ref.update({**creds, "credentials_rotated_at": _now()})
    return {"app_id": app_id, **creds}


def get_app_by_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Look up a client app by its API key."""
    if not api_key:
        return None
    db = get_firestore_client()
    docs = list(db.collection("payment_apps").where(filter=firestore.FieldFilter("api_key", "==", api_key.strip())).limit(1).stream())
    if docs:
        d = docs[0].to_dict()
        d["id"] = docs[0].id
        return d
    return None


def get_app(app_id: str) -> Optional[Dict[str, Any]]:
    if not app_id:
        return None
    doc = get_firestore_client().collection("payment_apps").document(app_id).get()
    return ({"id": doc.id, **(doc.to_dict() or {})}) if doc.exists else None


def list_client_apps() -> List[Dict[str, Any]]:
    """List all registered apps."""
    db = get_firestore_client()
    docs = db.collection("payment_apps").stream()
    return [{"id": d.id, **d.to_dict()} for d in docs]


def register_client_app(name: str, app_id: Optional[str] = None, webhook_url: str = "", description: str = "") -> Dict[str, Any]:
    """Register or update a client app with a unique API key."""
    db = get_firestore_client()
    if not app_id:
        app_id = name.lower().replace(" ", "-").replace("/", "-")
        app_id = "".join([c for c in app_id if c.isalnum() or c == "-"])

    data = {
        "app_id": app_id,
        "name": name,
        **_new_credentials(app_id),
        "webhook_url": webhook_url,
        "description": description,
        "is_active": True,
        "created_at": _now()
    }
    db.collection("payment_apps").document(app_id).set(data)
    return data


async def initiate_payment(
    app_id: str,
    amount_etb: float,
    currency: str = "ETB",
    tx_ref: Optional[str] = None,
    user_id: Optional[str] = None,
    email: Optional[str] = None,
    phone_number: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    callback_url: Optional[str] = None,
    return_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Central payment initiation.
    Calls Chapa initialize endpoint and saves the transaction in Firestore.
    """
    if not CHAPA_SECRET_KEY:
        raise GatewayError(503, "CHAPA_SECRET_KEY is not configured on My Portal")

    db = get_firestore_client()

    if tx_ref:
        if not TX_REF_PATTERN.match(tx_ref):
            raise GatewayError(400, "tx_ref may only contain letters, digits, '.', '_' and '-' (max 100)")
    else:
        tx_ref = f"{app_id}-{uuid.uuid4().hex[:12]}"

    effective_callback = callback_url or f"{MY_PORTAL_PUBLIC_URL}/api/v1/payments/webhook"
    effective_return = return_url or f"{MY_PORTAL_PUBLIC_URL}/payments"

    # Chapa payload
    chapa_payload = {
        "amount": str(int(amount_etb) if float(amount_etb).is_integer() else amount_etb),
        "currency": currency,
        "tx_ref": tx_ref,
        "callback_url": effective_callback,
        "return_url": effective_return,
        "email": email or f"customer-{user_id or 'anon'}@odaflux.com",
        "first_name": (first_name or "Valued")[:50],
        "last_name": (last_name or "Customer")[:50],
        "customization": {
            "title": f"Pay {app_id.upper()}"[:16],
            "description": f"Order {tx_ref}"[:50]
        }
    }
    if phone_number:
        chapa_payload["phone_number"] = phone_number

    tx_record = {
        "tx_ref": tx_ref,
        "app_id": app_id,
        "user_id": user_id or "anonymous",
        "amount_etb": float(amount_etb),
        "currency": currency,
        "status": "pending",
        "email": chapa_payload["email"],
        "phone_number": phone_number or "",
        "first_name": chapa_payload["first_name"],
        "last_name": chapa_payload["last_name"],
        "metadata": metadata or {},
        "client_callback_url": callback_url,
        "client_return_url": return_url,
        "created_at": _now(),
        "updated_at": _now()
    }
    doc_ref = db.collection("payment_transactions").document(tx_ref)
    # create() fails if the document exists. A caller-supplied tx_ref must never
    # overwrite an existing transaction, least of all another platform's.
    try:
        doc_ref.create(tx_record)
    except AlreadyExists:
        raise GatewayError(409, f"tx_ref {tx_ref} already exists")

    headers = {
        "Authorization": f"Bearer {CHAPA_SECRET_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{CHAPA_API_URL}/transaction/initialize", json=chapa_payload, headers=headers)
        try:
            data = resp.json()
        except ValueError:
            data = {}

        if resp.status_code != 200 or data.get("status") != "success":
            err_msg = data.get("message") or f"Chapa API HTTP {resp.status_code}"
            doc_ref.update({"status": "failed", "error": str(err_msg), "updated_at": _now()})
            raise GatewayError(502, f"Chapa initialization failed: {err_msg}")

        checkout_url = data.get("data", {}).get("checkout_url")
        doc_ref.update({"checkout_url": checkout_url, "updated_at": _now()})

        return {
            "status": "success",
            "message": "Payment initialized successfully",
            "tx_ref": tx_ref,
            "checkout_url": checkout_url,
            "app_id": app_id,
            "amount_etb": float(amount_etb)
        }


# ==================== Verification (source of truth) ====================

async def fetch_chapa_verification(tx_ref: str) -> Optional[Dict[str, Any]]:
    """Ask Chapa for the authoritative state of a transaction.

    Returns Chapa's `data` object, or None if Chapa could not be reached or did
    not return a usable answer.
    """
    if not CHAPA_SECRET_KEY:
        return None
    headers = {"Authorization": f"Bearer {CHAPA_SECRET_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{CHAPA_API_URL}/transaction/verify/{tx_ref}", headers=headers)
    except httpx.HTTPError as e:
        logger.error("Chapa verify request failed for %s: %s", tx_ref, e)
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json().get("data")
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def status_from_verification(record: Optional[Dict[str, Any]], vdata: Optional[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """Map Chapa's verified state onto the ledger status.

    Returns (status, error). Rules:
      * A completed payment is never moved back to pending or failed.
      * Chapa "success" whose amount or currency disagrees with what we asked
        for becomes "needs_review" rather than "completed".
      * Anything Chapa does not report as settled leaves the status alone.
    """
    current = (record or {}).get("status", "pending")
    if current == "completed" or not vdata:
        return current, None

    chapa_status = str(vdata.get("status") or "").strip().lower()

    if chapa_status == "success":
        if record:
            mismatch = _amount_mismatch(record, vdata)
            if mismatch:
                return "needs_review", mismatch
        return "completed", None

    if chapa_status in ("failed", "failure", "cancelled", "canceled"):
        return "failed", f"Chapa reported {chapa_status}"

    return current, None


def _amount_mismatch(record: Dict[str, Any], vdata: Dict[str, Any]) -> Optional[str]:
    try:
        expected = float(record.get("amount_etb"))
        paid = float(vdata.get("amount"))
    except (TypeError, ValueError):
        return None  # Chapa did not report a comparable amount; nothing to check.
    if abs(expected - paid) > 0.009:
        return f"amount mismatch: expected {expected:.2f}, Chapa reported {paid:.2f}"
    exp_cur = str(record.get("currency") or "ETB").upper()
    paid_cur = str(vdata.get("currency") or exp_cur).upper()
    if exp_cur != paid_cur:
        return f"currency mismatch: expected {exp_cur}, Chapa reported {paid_cur}"
    return None


async def verify_payment(tx_ref: str) -> Dict[str, Any]:
    """
    Verify payment status for a transaction.
    Checks Firestore and syncs directly with Chapa API if needed.
    """
    db = get_firestore_client()
    doc_ref = db.collection("payment_transactions").document(tx_ref)
    doc = doc_ref.get()
    record = doc.to_dict() if doc.exists else None

    vdata = await fetch_chapa_verification(tx_ref)
    final_status, error = status_from_verification(record, vdata)

    if record:
        update_data = {"status": final_status, "updated_at": _now()}
        if vdata is not None:
            update_data["chapa_verification"] = vdata
        if error:
            update_data["error"] = error
        doc_ref.update(update_data)
        record.update(update_data)
    elif vdata is None:
        final_status = "pending"
    else:
        final_status = "completed" if str(vdata.get("status", "")).lower() == "success" else str(vdata.get("status") or "pending").lower()

    return {
        "status": final_status,
        "tx_ref": tx_ref,
        "amount_etb": record.get("amount_etb") if record else (vdata or {}).get("amount"),
        "currency": "ETB",
        "app_id": record.get("app_id") if record else "unknown",
        "user_id": record.get("user_id") if record else None,
        "chapa_data": vdata
    }


# ==================== Webhook ====================

def check_webhook_signature(raw_body: bytes, headers) -> str:
    """Verify Chapa's HMAC signature over the raw request body.

    Chapa has documented two schemes (`Chapa-Signature` = HMAC(secret, body);
    `x-chapa-signature` = HMAC(secret, secret)), so both headers and both
    schemes are accepted; all of them require knowledge of the secret. The
    matched scheme is returned and logged so it can be pinned once observed.

    If CHAPA_WEBHOOK_SECRET is not configured the request is let through with a
    warning. That is safe only because nothing in a webhook body is trusted:
    the ledger is settled from Chapa's verify API, never from the payload.
    """
    if not CHAPA_WEBHOOK_SECRET:
        logger.warning("CHAPA_WEBHOOK_SECRET not set: webhook accepted without signature check "
                       "(status still comes from Chapa verify)")
        return "unverified:no-secret-configured"

    presented = [(h, headers.get(h).strip()) for h in SIGNATURE_HEADERS if headers.get(h)]
    if not presented:
        raise GatewayError(401, "Missing Chapa signature header")

    key = CHAPA_WEBHOOK_SECRET.encode()
    candidates = {
        "payload-hmac": hmac.new(key, raw_body, hashlib.sha256).hexdigest(),
        "secret-hmac": hmac.new(key, key, hashlib.sha256).hexdigest(),
    }
    for header_name, value in presented:
        for scheme, expected in candidates.items():
            if hmac.compare_digest(value.encode("utf-8", "replace"), expected.encode("ascii")):
                return f"{header_name}:{scheme}"
    raise GatewayError(401, "Invalid Chapa webhook signature")


def _extract_tx_ref(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if isinstance(data, dict) and data.get("tx_ref"):
        return str(data["tx_ref"])
    for key in ("tx_ref", "trx_ref"):
        if payload.get(key):
            return str(payload[key])
    return None


async def handle_chapa_webhook(raw_body: bytes, headers) -> Dict[str, Any]:
    """
    Process an incoming Chapa webhook.

    The body is used only to learn WHICH transaction changed. Its status is
    re-read from Chapa's verify API, so a forged or replayed webhook cannot
    mark a payment completed. Successful payments are forwarded to the owning
    platform exactly once.
    """
    scheme = check_webhook_signature(raw_body, headers)

    try:
        payload = json.loads(raw_body or b"{}")
    except (ValueError, UnicodeDecodeError):
        raise GatewayError(400, "Invalid JSON payload")

    tx_ref = _extract_tx_ref(payload)
    if not tx_ref or not TX_REF_PATTERN.match(tx_ref):
        raise GatewayError(400, "Missing or invalid tx_ref in webhook payload")

    db = get_firestore_client()
    doc_ref = db.collection("payment_transactions").document(tx_ref)
    doc = doc_ref.get()
    if not doc.exists:
        # Do not create records from webhook input. 200 so Chapa stops retrying.
        logger.warning("Webhook for unknown tx_ref %s ignored", tx_ref)
        return {"status": "ignored", "tx_ref": tx_ref, "reason": "unknown tx_ref"}

    record = doc.to_dict() or {}
    vdata = await fetch_chapa_verification(tx_ref)
    if vdata is None:
        # 502 makes Chapa retry; better than settling on an unverified claim.
        raise GatewayError(502, "Could not verify transaction with Chapa")

    new_status, error = status_from_verification(record, vdata)
    update = {
        "status": new_status,
        "webhook_payload": payload,
        "webhook_signature_scheme": scheme,
        "chapa_verification": vdata,
        "updated_at": _now(),
    }
    if error:
        update["error"] = error
    doc_ref.update(update)
    record.update(update)

    forward_status = record.get("forward_status", "none")
    if new_status == "completed" and not record.get("forwarded_at"):
        forward_status, delivered = await _forward_success(db, tx_ref, record, vdata)
        forward_update = {"forward_status": forward_status}
        if delivered:
            forward_update["forwarded_at"] = _now()
        doc_ref.update(forward_update)

    return {"status": new_status, "tx_ref": tx_ref, "forward": forward_status}


async def _forward_success(db, tx_ref: str, record: Dict[str, Any], vdata: Dict[str, Any]) -> Tuple[str, bool]:
    """Send payment.success to the owning platform. Returns (status text, delivered)."""
    app_id = record.get("app_id")
    if not app_id:
        return "none", False
    app_doc = db.collection("payment_apps").document(app_id).get()
    if not app_doc.exists:
        return "none", False
    app_info = app_doc.to_dict() or {}
    target_url = app_info.get("webhook_url")
    if not target_url:
        return "none", False

    forward_data = {
        "event": "payment.success",
        "tx_ref": tx_ref,
        "app_id": app_id,
        "user_id": record.get("user_id"),
        "amount_etb": record.get("amount_etb"),
        "status": "completed",
        "metadata": record.get("metadata", {}),
        "chapa_data": vdata
    }
    raw_json = json.dumps(forward_data, separators=(',', ':'))
    secret = app_info.get("webhook_secret") or app_info.get("api_key") or ""
    signature = hmac.new(secret.encode('utf-8'), raw_json.encode('utf-8'), hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Portal-Signature": signature,
        "X-Portal-App-Id": app_id,
        "X-Portal-Forward": "true",
        # Kept for receivers that compare the shared secret directly (see commit
        # 0041982). Receivers should verify X-Portal-Signature instead; once they
        # all do, drop this header so the secret is not sent on every request.
        "X-Webhook-Secret": secret
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            f_resp = await client.post(target_url, content=raw_json, headers=headers)
        return f"forwarded (HTTP {f_resp.status_code})", 200 <= f_resp.status_code < 300
    except Exception as e:
        return f"forward_error: {str(e)}", False


# ==================== Stats ====================

STAT_STATUSES = ("completed", "pending", "failed", "needs_review")


def _blank_bucket() -> Dict[str, Any]:
    return {
        "total_transactions": 0,
        "completed_count": 0,
        "pending_count": 0,
        "failed_count": 0,
        "needs_review_count": 0,
        "completed_etb": 0.0,
        "pending_etb": 0.0,
    }


def _bucket_status(status: Any) -> str:
    s = str(status or "pending").lower()
    return s if s in STAT_STATUSES else "pending"


def _rate(completed: int, total: int) -> Optional[str]:
    return f"{(completed / total * 100):.1f}%" if total > 0 else None


def compute_payment_stats(transactions: Iterable[Dict[str, Any]], apps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate transactions into overall and per-platform totals.

    Every registered app appears in `by_app`, even with no transactions, so a
    quiet platform shows as zero rather than disappearing. Transactions whose
    app_id is not registered are grouped under their own id ("unassigned" if
    missing).
    """
    names = {a.get("app_id") or a.get("id"): a.get("name") for a in apps}
    buckets: Dict[str, Dict[str, Any]] = {app_id: _blank_bucket() for app_id in names if app_id}
    total = _blank_bucket()

    for t in transactions:
        app_id = t.get("app_id") or "unassigned"
        status = _bucket_status(t.get("status"))
        try:
            amount = float(t.get("amount_etb") or 0)
        except (TypeError, ValueError):
            amount = 0.0
        for b in (buckets.setdefault(app_id, _blank_bucket()), total):
            b["total_transactions"] += 1
            b[f"{status}_count"] += 1
            if status == "completed":
                b["completed_etb"] += amount
            elif status == "pending":
                b["pending_etb"] += amount

    by_app = []
    for app_id, b in buckets.items():
        by_app.append({
            "app_id": app_id,
            "name": names.get(app_id) or app_id,
            **b,
            "completed_etb": round(b["completed_etb"], 2),
            "pending_etb": round(b["pending_etb"], 2),
            "success_rate": _rate(b["completed_count"], b["total_transactions"]),
        })
    by_app.sort(key=lambda r: (-r["completed_etb"], r["app_id"]))

    return {
        # Unchanged fields the dashboard already reads.
        "total_volume_etb": round(total["completed_etb"], 2),
        "total_transactions": total["total_transactions"],
        "completed_count": total["completed_count"],
        "pending_count": total["pending_count"],
        "failed_count": total["failed_count"],
        "success_rate": _rate(total["completed_count"], total["total_transactions"]) or "100%",
        "apps_count": len(apps),
        # New.
        "needs_review_count": total["needs_review_count"],
        "pending_volume_etb": round(total["pending_etb"], 2),
        "by_app": by_app,
    }
