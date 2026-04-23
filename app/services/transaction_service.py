"""
Musa AI Transaction Service - Integer Credits Only
Handles atomic credit deductions with proper state management.
"""

from datetime import datetime
from typing import Dict, Any, Optional


class TransactionService:
    """
    Transaction service using INTEGER credits only.
    All credit values are integers - no floating point.
    """
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
    
    async def process_transaction(
        self, 
        user_id: str, 
        total_cost: int,           # ✅ Integer only
        model_id: str,
        base_cost: int,            # ✅ Integer only
        template_id: Optional[str] = None,
        creator_id: Optional[str] = None,
        creator_premium: int = 0,  # ✅ Integer only
        description: str = "Generation"
    ) -> Dict[str, Any]:
        """
        Process a transaction using atomic database RPC.
        
        All credit values must be integers.
        
        Args:
            user_id: User UUID
            total_cost: Total credits to deduct (INTEGER)
            model_id: AI model used
            base_cost: Base model cost (INTEGER)
            template_id: Optional template used
            creator_id: Optional creator for revenue split
            creator_premium: Creator fee amount (INTEGER)
            description: Transaction description
        
        Returns:
            {
                "transaction_id": "uuid",
                "new_balance": 100,
                "status": "success"
            }
        
        Raises:
            ValueError: If insufficient funds or other error
            Exception: If transaction processing fails
        """
        
        # Validate all amounts are integers
        if not all(isinstance(x, int) for x in [total_cost, base_cost, creator_premium]):
            raise ValueError("All credit amounts must be integers")
        
        # Validate amounts are non-negative
        if total_cost < 0 or base_cost < 0 or creator_premium < 0:
            raise ValueError("Credit amounts cannot be negative")
        
        # Call the PostgreSQL RPC function for atomic transaction
        result = self.supabase.rpc(
            "process_musa_transaction",
            {
                "p_user_id": user_id,
                "p_total_cost": total_cost,
                "p_model_id": model_id,
                "p_template_id": template_id,
                "p_creator_id": creator_id,
                "p_creator_premium": creator_premium,
                "p_base_cost": base_cost,
                "p_description": description
            }
        ).execute()
        
        if not result.data or len(result.data) == 0:
            raise Exception("Transaction processing failed: No data returned from database")
        
        response = result.data[0]
        
        # Check for errors returned from the function
        if response.get("error_message"):
            raise ValueError(response["error_message"])
        
        return {
            "transaction_id": response["transaction_id"],
            "new_balance": int(response["new_balance"]),  # Ensure integer
            "status": "success"
        }
    
    async def process_refund(
        self,
        original_transaction_id: str,
        user_id: str,
        amount: int,
        reason: str
    ) -> Dict[str, Any]:
        """
        Process a refund for a failed generation.
        
        Args:
            original_transaction_id: ID of original transaction
            user_id: User to refund
            amount: Credits to refund (INTEGER)
            reason: Refund reason
        
        Returns:
            Refund transaction details
        """
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("Refund amount must be a positive integer")
        
        # Create refund transaction record
        result = self.supabase.table("transactions").insert({
            "user_id": user_id,
            "type": "refund",
            "amount": amount,
            "description": f"Refund: {reason}",
            "reference": original_transaction_id,
            "status": "completed",
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        if not result.data:
            raise Exception("Failed to create refund record")
        
        # Update user balance
        balance_result = self.supabase.rpc(
            "add_user_credits",
            {"p_user_id": user_id, "p_credits": amount}
        ).execute()
        
        if not balance_result.data:
            raise Exception("Failed to update user balance")
        
        return {
            "refund_id": result.data[0]["id"],
            "amount_refunded": amount,
            "new_balance": int(balance_result.data[0]["new_balance"]),
            "status": "completed"
        }
    
    async def mark_for_investigation(
        self,
        user_id: str,
        model_id: str,
        prompt: str,
        transaction_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Mark a transaction for manual investigation.
        Used for timeouts and ambiguous failures.
        
        These are NOT auto-refunded - requires manual review.
        """
        result = self.supabase.table("investigation_queue").insert({
            "user_id": user_id,
            "model_id": model_id,
            "prompt_preview": prompt[:100],
            "transaction_id": transaction_id,
            "reason": reason,
            "status": "pending_investigation",
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        return {
            "investigation_id": result.data[0]["id"] if result.data else None,
            "status": "pending_investigation",
            "message": "Your request is being investigated. Do not retry immediately."
        }
    
    async def get_transaction_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get user's transaction history."""
        result = self.supabase.table("transactions").select(
            "*"
        ).eq("user_id", user_id).order(
            "created_at", desc=True
        ).range(offset, offset + limit - 1).execute()
        
        return {
            "transactions": result.data or [],
            "count": len(result.data) if result.data else 0
        }
