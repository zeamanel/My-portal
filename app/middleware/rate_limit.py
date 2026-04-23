"""
Rate Limiting Middleware for Musa AI
Prevents API abuse and controls costs.
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
import redis
import time


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting per user/API key.
    
    Default limits:
    - 100 requests per minute per user
    - 50 generations per hour per user
    - 3 Chapa initiations per hour per user
    """
    
    def __init__(
        self, 
        app, 
        redis_client: redis.Redis,
        requests_per_minute: int = 100,
        generation_limit_per_hour: int = 50,
        chapa_limit_per_hour: int = 3
    ):
        super().__init__(app)
        self.redis = redis_client
        self.requests_per_minute = requests_per_minute
        self.generation_limit_per_hour = generation_limit_per_hour
        self.chapa_limit_per_hour = chapa_limit_per_hour
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and static files
        if self._should_skip_rate_limit(request):
            return await call_next(request)
        
        # Get user identifier
        user_id = await self._get_user_id(request)
        
        if not user_id:
            # Rate limit by IP for unauthenticated requests (stricter)
            user_id = f"ip:{request.client.host}"
            is_authenticated = False
        else:
            is_authenticated = True
        
        # Check general rate limit
        if not await self._check_general_rate_limit(user_id, is_authenticated):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again in a minute.",
                headers={"Retry-After": "60"}
            )
        
        # Special limits for specific endpoints
        if "/generate" in request.url.path and request.method == "POST":
            if not await self._check_generation_limit(user_id):
                raise HTTPException(
                    status_code=429,
                    detail="Generation limit exceeded. Max 50 per hour.",
                    headers={"Retry-After": str(self._get_generation_retry_after(user_id))}
                )
        
        if "/chapa/initiate" in request.url.path:
            if not await self._check_chapa_limit(user_id):
                raise HTTPException(
                    status_code=429,
                    detail="Payment initiation limit exceeded. Max 3 per hour.",
                    headers={"Retry-After": str(self._get_chapa_retry_after(user_id))}
                )
        
        return await call_next(request)
    
    def _should_skip_rate_limit(self, request: Request) -> bool:
        """Check if request should skip rate limiting."""
        path = request.url.path
        
        # Skip health checks
        if path in ["/health", "/", "/docs", "/openapi.json"]:
            return True
        
        # Skip webhook endpoints (they have their own auth)
        if "/webhook" in path:
            return True
        
        return False
    
    async def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request (JWT token or API key)."""
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return None
        
        # Extract from Bearer token (simplified)
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # In production, decode JWT to get user_id
            # For now, return token prefix as identifier
            return f"user:{token[:16]}"
        
        return None
    
    async def _check_general_rate_limit(
        self, 
        user_id: str, 
        is_authenticated: bool
    ) -> bool:
        """Check general API rate limit."""
        # Stricter limits for unauthenticated requests
        limit = self.requests_per_minute if is_authenticated else 20
        
        key = f"rate_limit:general:{user_id}"
        current = self.redis.get(key)
        
        if not current:
            self.redis.setex(key, 60, 1)
            return True
        
        count = int(current)
        if count >= limit:
            return False
        
        self.redis.incr(key)
        return True
    
    async def _check_generation_limit(self, user_id: str) -> bool:
        """Check generation-specific rate limit."""
        key = f"rate_limit:generation:{user_id}"
        current = self.redis.get(key)
        
        if not current:
            self.redis.setex(key, 3600, 1)  # 1 hour
            return True
        
        count = int(current)
        if count >= self.generation_limit_per_hour:
            return False
        
        self.redis.incr(key)
        return True
    
    async def _check_chapa_limit(self, user_id: str) -> bool:
        """Check Chapa payment initiation limit."""
        key = f"rate_limit:chapa:{user_id}"
        current = self.redis.get(key)
        
        if not current:
            self.redis.setex(key, 3600, 1)  # 1 hour
            return True
        
        count = int(current)
        if count >= self.chapa_limit_per_hour:
            return False
        
        self.redis.incr(key)
        return True
    
    def _get_generation_retry_after(self, user_id: str) -> int:
        """Get seconds until generation limit resets."""
        key = f"rate_limit:generation:{user_id}"
        ttl = self.redis.ttl(key)
        return max(ttl, 60) if ttl > 0 else 3600
    
    def _get_chapa_retry_after(self, user_id: str) -> int:
        """Get seconds until Chapa limit resets."""
        key = f"rate_limit:chapa:{user_id}"
        ttl = self.redis.ttl(key)
        return max(ttl, 60) if ttl > 0 else 3600
