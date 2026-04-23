"""
Chapa Payment Integration for Musa AI
Handles ETB currency payments with webhook verification.
"""

import hmac
import hashlib
import os
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import HTTPException, Header, Request
import httpx
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://www.odaflux.com")

class ChapaIntegration:
    """
    Chapa Payment Integration for ETB currency.
    Server-side only - never expose API keys to frontend.
    """
    
    CHAPA_API_URL = "https://api.chapa.co/v1"
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.secret_key = os.getenv("CHAPA_SECRET_KEY")
        self.public_key = os.getenv("CHAPA_PUBLIC_KEY")
        self.webhook_secret = os.getenv("CHAPA_WEBHOOK_SECRET")
        
        if not all([self.secret_key, self.public_key, self.webhook_secret]):
            raise ValueError("Chapa credentials not configured")
    
    async def initiate_payment(
        self, 
        user_id: str, 
        amount_etb: int,
        phone_number: str,
        callback_url: str
    ) -> Dict[str, Any]:
        """
        Initiate Chapa payment.
        
        Rate limit: Max 3 initiations per hour per user.
        Amount limits: MIN=10 ETB, MAX=10000 ETB
        
        Returns:
            {
                "checkout_url": "https://checkout.chapa.co/...",
                "tx_ref": "musa-uuid-timestamp",
                "status": "pending"
            }
        """
        # Rate limit check
        if not await self._check_rate_limit(user_id):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Max 3 payment initiations per hour."
            )
        
        # Validate amount
        if amount_etb < 10:
            raise HTTPException(
                status_code=400,
                detail="Minimum amount is 10 ETB"
            )
        
        if amount_etb > 10000:
            raise HTTPException(
                status_code=400,
                detail="Maximum amount is 10000 ETB"
            )
        
        # Generate unique transaction reference (short enough for Chapa)
        tx_ref = f"tx-{uuid.uuid4().hex[:10]}-{int(time.time()*1000)}"
        
        # Create pending transaction record
        await self._create_pending_transaction(user_id, tx_ref, amount_etb)
        
        # Call Chapa API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.CHAPA_API_URL}/transaction/initialize",
                headers={"Authorization": f"Bearer {self.secret_key}"},
                json={
                    "amount": str(amount_etb),
                    "currency": "ETB",
                    "phone_number": phone_number,
                    "tx_ref": tx_ref,
                    "callback_url": callback_url,
                    "return_url": f"{FRONTEND_URL}/payment-success"
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                # Update transaction as failed
                self.supabase.table("chapa_transactions").update({
                    "status": "failed",
                    "failure_reason": f"Chapa API error: {response.text}"
                }).eq("tx_ref", tx_ref).execute()
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Chapa API error: {response.text}"
                )
            
            data = response.json()
            
            if data.get("status") != "success":
                raise HTTPException(
                    status_code=500,
                    detail=f"Chapa initialization failed: {data.get('message')}"
                )
            
            return {
                "checkout_url": data["data"]["checkout_url"],
                "tx_ref": tx_ref,
                "status": "pending"
            }
    
    async def verify_webhook(self, request: Request) -> Dict[str, Any]:
        """
        Verify Chapa webhook signature.
        Extracts signature from request headers manually.
        """
        body = await request.body()
        x_chapa_signature = request.headers.get("x-chapa-signature")
    
        if not x_chapa_signature:
            raise HTTPException(
                status_code=401,
                detail="Missing X-Chapa-Signature header"
            )
    
        expected_signature = hmac.new(
            self.webhook_secret.encode('utf-8'),
            body,
            hashlib.sha256
        ).hexdigest()
    
        if not hmac.compare_digest(x_chapa_signature, expected_signature):
            self._log_security_event(
                event_type="invalid_webhook_signature",
                details={"received": x_chapa_signature[:20] + "..."}
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid signature"
            )
    
        return await request.json()
    
    async def process_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process verified Chapa webhook.
        
        This function is idempotent - safe to call multiple times
        with the same transaction reference.
        
        Args:
            webhook_data: Verified webhook data from Chapa
        
        Returns:
            Processing result
        """
        tx_ref = webhook_data.get("tx_ref")
        status = webhook_data.get("status")  # success / failed / pending
        
        if not tx_ref:
            raise HTTPException(status_code=400, detail="Missing tx_ref")
        
        # Idempotency check - already processed?
        existing = self.supabase.table("chapa_transactions").select(
            "status", "credits_added"
        ).eq("tx_ref", tx_ref).execute()
        
        if existing.data:
            tx_record = existing.data[0]
            if tx_record["status"] == "completed":
                return {
                    "status": "already_processed",
                    "tx_ref": tx_ref,
                    "credits_added": tx_record.get("credits_added", 0)
                }
        
        if status == "success":
            return await self._process_successful_payment(tx_ref, webhook_data)
        else:
            return await self._process_failed_payment(tx_ref, webhook_data)
    
    async def _process_successful_payment(
        self, 
        tx_ref: str, 
        webhook_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process successful payment and add credits."""
        
        # Get pending transaction
        pending = self.supabase.table("chapa_transactions").select(
            "*"
        ).eq("tx_ref", tx_ref).eq("status", "pending").execute()
        
        if not pending.data:
            raise HTTPException(
                status_code=404, 
                detail="Pending transaction not found"
            )
        
        pending_tx = pending.data[0]
        user_id = pending_tx["user_id"]
        amount_etb = pending_tx["amount_etb"]
        
        # Calculate credits (1 ETB = 10 credits)
        credits_to_add = amount_etb * 10
        
        # Atomic credit update using Supabase RPC
        result = self.supabase.rpc(
            "add_user_credits",
            {"p_user_id": user_id, "p_credits": credits_to_add}
        ).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=500, 
                detail="Failed to add credits"
            )
        
        new_balance = result.data[0]["new_balance"]
        
        # Update transaction status
        self.supabase.table("chapa_transactions").update({
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat(),
            "chapa_transaction_id": webhook_data.get("id"),
            "credits_added": credits_to_add
        }).eq("tx_ref", tx_ref).execute()
        
        # Log successful payment
        print(f"✅ Chapa payment completed: {tx_ref}, Credits added: {credits_to_add}")
        
        return {
            "status": "success",
            "tx_ref": tx_ref,
            "credits_added": credits_to_add,
            "new_balance": new_balance
        }
    
    async def _process_failed_payment(
        self, 
        tx_ref: str, 
        webhook_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process failed payment."""
        
        self.supabase.table("chapa_transactions").update({
            "status": "failed",
            "failure_reason": webhook_data.get("message", "Unknown"),
            "processed_at": datetime.utcnow().isoformat()
        }).eq("tx_ref", tx_ref).execute()
        
        print(f"❌ Chapa payment failed: {tx_ref}")
        
        return {
            "status": "failed",
            "tx_ref": tx_ref,
            "reason": webhook_data.get("message", "Unknown")
        }
    
    async def _check_rate_limit(self, user_id: str) -> bool:
        """
        Check if user has exceeded rate limit.
        Max 3 payment initiations per hour per user.
        """
        one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        
        result = self.supabase.table("chapa_transactions").select(
            "count", count="exact"
        ).eq("user_id", user_id).gte("created_at", one_hour_ago).execute()
        
        count = result.count if hasattr(result, 'count') else len(result.data or [])
        return count < 3
    
    async def _create_pending_transaction(
        self, 
        user_id: str, 
        tx_ref: str, 
        amount: int
    ):
        """Create pending transaction record."""
        self.supabase.table("chapa_transactions").insert({
            "user_id": user_id,
            "tx_ref": tx_ref,
            "amount_etb": amount,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }).execute()
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for monitoring."""
        print(f"🚨 SECURITY EVENT: {event_type} - {details}")
        # In production, send to security monitoring (SIEM)