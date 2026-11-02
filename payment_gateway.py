import os
import hmac
import hashlib
import json
import uuid
import datetime
import secrets
from typing import Optional, Dict, Any, List
import httpx
from google.cloud import firestore
from gcp_client import get_firestore_client

CHAPA_SECRET_KEY = os.getenv("CHAPA_SECRET_KEY", "CHASECK-h0e7lzSNPvueyMT3cIQyXmef6o1vDvFq")
CHAPA_PUBLIC_KEY = os.getenv("CHAPA_PUBLIC_KEY", "CHAPUBK-aOfZO6EM1GDKJtPfyNq6EHhNSuiwE61n")
CHAPA_WEBHOOK_SECRET = os.getenv("CHAPA_WEBHOOK_SECRET", "1234567890")
CHAPA_API_URL = "https://api.chapa.co/v1"
MY_PORTAL_PUBLIC_URL = os.getenv("PUBLIC_URL", "https://my-portal-1045664496839.us-central1.run.app")

# Default 4 registered apps
DEFAULT_APPS = [
    {
        "app_id": "musa-ai",
        "name": "Musa AI / Odaflux",
        "api_key": "chp_live_musa_ai_8f3d120a9c4b",
        "webhook_secret": "whsec_live_musa_ai_19f8b2d4",
        "webhook_url": "https://musa-ai-backend-1045664496839.us-central1.run.app/api/v1/chapa/webhook",
        "description": "Image/Video generation platform credits recharge",
        "is_active": True
    },
    {
        "app_id": "ethiogram",
        "name": "Ethiogram",
        "api_key": "chp_live_ethiogram_4e9b210c8a7f",
        "webhook_secret": "whsec_live_ethiogram_8b7d42a1",
        "webhook_url": "https://ethiogram.com/api/billing/chapa-webhook",
        "description": "Ethiogram Telegram marketing platform balance",
        "is_active": True
    },
    {
        "app_id": "shaleqa",
        "name": "Shaleqa",
        "api_key": "chp_live_shaleqa_7b1d439a2e5c",
        "webhook_secret": "whsec_live_shaleqa_5c3a90e2",
        "webhook_url": "https://shaleqa-backend-1045664496839.us-central1.run.app/api/v1/payments/chapa-callback",
        "description": "Shaleqa marketplace orders and customer payments",
        "is_active": True
    },
    {
        "app_id": "system-backend",
        "name": "System Backend",
        "api_key": "chp_live_sys_backend_9c8a143f5d2b",
        "webhook_secret": "whsec_live_sys_backend_e4b1029c",
        "webhook_url": "https://systembackendservice-ebjq5oaqna-uc.a.run.app/api/v1/payments/callback",
        "description": "Internal business workflows & vendor settlements",
        "is_active": True
    }
]

def ensure_default_apps():
    """Seed or update the 4 default client apps in Firestore."""
    db = get_firestore_client()
    coll = db.collection("payment_apps")
    for app in DEFAULT_APPS:
        doc = coll.document(app["app_id"]).get()
        if not doc.exists:
            app_data = dict(app)
            app_data["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            coll.document(app["app_id"]).set(app_data)
            print(f"  + Registered client app: {app['name']} (Key: {app['api_key']})")
        else:
            # Update webhook_secret if missing
            doc_data = doc.to_dict() or {}
            if not doc_data.get("webhook_secret"):
                coll.document(app["app_id"]).update({
                    "webhook_secret": app["webhook_secret"]
                })

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
    
    api_key = f"chp_live_{app_id[:12]}_{secrets.token_hex(6)}"
    webhook_secret = f"whsec_live_{app_id[:12]}_{secrets.token_hex(6)}"
    data = {
        "app_id": app_id,
        "name": name,
        "api_key": api_key,
        "webhook_secret": webhook_secret,
        "webhook_url": webhook_url,
        "description": description,
        "is_active": True,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
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
        raise ValueError("CHAPA_SECRET_KEY is not configured on My Portal")

    db = get_firestore_client()

    # Generate unique tx_ref if not supplied
    if not tx_ref:
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

    # Save transaction record in Firestore
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
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    db.collection("payment_transactions").document(tx_ref).set(tx_record)

    headers = {
        "Authorization": f"Bearer {CHAPA_SECRET_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{CHAPA_API_URL}/transaction/initialize", json=chapa_payload, headers=headers)
        data = resp.json()

        if resp.status_code != 200 or data.get("status") != "success":
            err_msg = data.get("message") or f"Chapa API HTTP {resp.status_code}"
            db.collection("payment_transactions").document(tx_ref).update({
                "status": "failed",
                "error": err_msg,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })
            raise RuntimeError(f"Chapa initialization failed: {err_msg}")

        checkout_url = data.get("data", {}).get("checkout_url")
        db.collection("payment_transactions").document(tx_ref).update({
            "checkout_url": checkout_url,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

        return {
            "status": "success",
            "message": "Payment initialized successfully",
            "tx_ref": tx_ref,
            "checkout_url": checkout_url,
            "app_id": app_id,
            "amount_etb": float(amount_etb)
        }

async def verify_payment(tx_ref: str) -> Dict[str, Any]:
    """
    Verify payment status for a transaction.
    Checks Firestore and syncs directly with Chapa API if needed.
    """
    db = get_firestore_client()
    doc_ref = db.collection("payment_transactions").document(tx_ref)
    doc = doc_ref.get()

    record = doc.to_dict() if doc.exists else None

    # Query Chapa API directly to verify
    headers = {"Authorization": f"Bearer {CHAPA_SECRET_KEY}"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{CHAPA_API_URL}/transaction/verify/{tx_ref}", headers=headers)
        chapa_data = resp.json() if resp.status_code == 200 else {}

    chapa_status = chapa_data.get("data", {}).get("status", "").lower()
    final_status = "completed" if chapa_status == "success" else (chapa_status or (record.get("status") if record else "pending"))

    if record:
        update_data = {
            "status": final_status,
            "chapa_verification": chapa_data.get("data"),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        doc_ref.update(update_data)
        record.update(update_data)

    return {
        "status": final_status,
        "tx_ref": tx_ref,
        "amount_etb": record.get("amount_etb") if record else chapa_data.get("data", {}).get("amount"),
        "currency": "ETB",
        "app_id": record.get("app_id") if record else "unknown",
        "user_id": record.get("user_id") if record else None,
        "chapa_data": chapa_data.get("data")
    }

async def handle_chapa_webhook(payload: Dict[str, Any], signature: Optional[str] = None) -> Dict[str, Any]:
    """
    Process incoming Chapa webhook, mark transaction as completed in Firestore,
    and forward the confirmation event to the specific client webapp.
    """
    # 1. Signature check if secret is configured
    if signature and CHAPA_WEBHOOK_SECRET and CHAPA_WEBHOOK_SECRET != "1234567890":
        expected = hmac.new(CHAPA_WEBHOOK_SECRET.encode(), json.dumps(payload).encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid Chapa webhook signature")

    tx_ref = payload.get("tx_ref")
    if not tx_ref:
        raise ValueError("Missing tx_ref in webhook payload")

    db = get_firestore_client()
    doc_ref = db.collection("payment_transactions").document(tx_ref)
    doc = doc_ref.get()

    if not doc.exists:
        # Save received webhook anyway for record
        doc_ref.set({
            "tx_ref": tx_ref,
            "status": "completed" if payload.get("status") == "success" else "failed",
            "webhook_payload": payload,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
        record = doc_ref.get().to_dict()
    else:
        record = doc.to_dict()
        doc_ref.update({
            "status": "completed" if payload.get("status") == "success" else "failed",
            "webhook_payload": payload,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

    # Forward to client app's webhook_url if registered
    app_id = record.get("app_id")
    forward_status = "none"
    if app_id:
        app_doc = db.collection("payment_apps").document(app_id).get()
        if app_doc.exists:
            app_info = app_doc.to_dict() or {}
            target_url = app_info.get("webhook_url")
            if target_url:
                forward_data = {
                    "event": "payment.success",
                    "tx_ref": tx_ref,
                    "app_id": app_id,
                    "user_id": record.get("user_id"),
                    "amount_etb": record.get("amount_etb"),
                    "status": "completed",
                    "metadata": record.get("metadata", {}),
                    "chapa_data": payload
                }
                raw_json = json.dumps(forward_data, separators=(',', ':'))
                # Compute HMAC signature with webhook_secret (or fallback to api_key)
                secret = app_info.get("webhook_secret") or app_info.get("api_key") or "secret"
                signature = hmac.new(secret.encode('utf-8'), raw_json.encode('utf-8'), hashlib.sha256).hexdigest()

                headers = {
                    "Content-Type": "application/json",
                    "X-Portal-Signature": signature,
                    "X-Portal-App-Id": app_id,
                    "X-Portal-Forward": "true",
                    "X-Webhook-Secret": secret
                }

                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        f_resp = await client.post(
                            target_url,
                            content=raw_json,
                            headers=headers
                        )
                        forward_status = f"forwarded (HTTP {f_resp.status_code})"
                except Exception as e:
                    forward_status = f"forward_error: {str(e)}"

    doc_ref.update({"forward_status": forward_status})
    return {"status": "ok", "tx_ref": tx_ref, "forward": forward_status}
