# app/services/chapa_service.py
import os
import uuid
import logging
import httpx
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

CHAPA_API_URL = "https://api.chapa.co/v1"


class ChapaService:
    def __init__(self) -> None:
        self.secret_key = os.getenv("CHAPA_SECRET_KEY")
        if not self.secret_key:
            raise ValueError("CHAPA_SECRET_KEY not configured")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.secret_key}",
            "Content-Type": "application/json",
        }

    async def initiate(
        self,
        *,
        tx_ref: str,
        amount_etb: float,
        email: str,
        first_name: str,
        last_name: str,
        phone_number: Optional[str],
        callback_url: str,
        return_url: str,
        title: str,
        description: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a Chapa checkout. Returns {"checkout_url": ..., "raw": {...}}."""
        payload = {
            "amount": f"{amount_etb:.2f}",
            "currency": "ETB",
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
            "tx_ref": tx_ref,
            "callback_url": callback_url,
            "return_url": return_url,
            "customization": {"title": title, "description": description},
            "meta": metadata,
        }
        if phone_number:
            payload["phone_number"] = phone_number

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{CHAPA_API_URL}/transaction/initialize",
                json=payload,
                headers=self._headers(),
            )
        if r.status_code != 200:
            logger.error("Chapa initiate failed %s: %s", r.status_code, r.text[:300])
            raise RuntimeError(f"Chapa initiate failed ({r.status_code})")

        data = r.json()
        checkout = data.get("data", {}).get("checkout_url")
        if not checkout:
            raise RuntimeError(f"Chapa initiate returned no checkout_url: {data}")
        return {"checkout_url": checkout, "raw": data}

    async def verify(self, tx_ref: str) -> Dict[str, Any]:
        """Verify a payment. Returns {"status": "success|failed|pending", "raw": {...}}."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                f"{CHAPA_API_URL}/transaction/verify/{tx_ref}",
                headers={"Authorization": f"Bearer {self.secret_key}"},
            )
        if r.status_code != 200:
            logger.error("Chapa verify failed %s for %s: %s", r.status_code, tx_ref, r.text[:300])
            raise RuntimeError(f"Chapa verify failed ({r.status_code})")
        data = r.json()
        return {
            "status": data.get("data", {}).get("status", "pending"),
            "raw": data,
        }

    async def refund(self, tx_ref: str, amount_etb: Optional[float] = None) -> Dict[str, Any]:
        """Issue a refund. `amount_etb=None` refunds the full amount."""
        payload: Dict[str, Any] = {"tx_ref": tx_ref}
        if amount_etb is not None:
            payload["amount"] = f"{amount_etb:.2f}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{CHAPA_API_URL}/refund",
                json=payload,
                headers=self._headers(),
            )
        if r.status_code not in (200, 201):
            logger.error("Chapa refund failed %s for %s: %s", r.status_code, tx_ref, r.text[:300])
            raise RuntimeError(f"Chapa refund failed ({r.status_code})")
        return r.json()
