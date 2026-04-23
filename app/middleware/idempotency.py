"""
Idempotency Middleware for Musa AI
Prevents duplicate transactions from double-clicks or retries.
"""

import json
import uuid
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis
from typing import Optional


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """
    Prevents duplicate requests using idempotency keys.
    
    How it works:
    1. Client sends X-Idempotency-Key header for POST/PUT/PATCH requests
    2. Server checks if key was already processed
    3. If yes, return cached response
    4. If no, process request and cache response
    
    Key TTL: 24 hours
    """
    
    def __init__(
        self, 
        app, 
        redis_client: redis.Redis,
        key_ttl_seconds: int = 86400  # 24 hours
    ):
        super().__init__(app)
        self.redis = redis_client
        self.key_ttl = key_ttl_seconds
    
    async def dispatch(self, request: Request, call_next):
        # Only check for state-changing requests
        if request.method not in ["POST", "PUT", "PATCH"]:
            return await call_next(request)
        
        # Skip idempotency for certain endpoints
        if self._should_skip_idempotency(request):
            return await call_next(request)
        
        # Get idempotency key from header
        idempotency_key = request.headers.get("X-Idempotency-Key")
        
        if not idempotency_key:
            # Generate one for tracking (optional - can require client to provide)
            idempotency_key = str(uuid.uuid4())
            request.state.idempotency_key = idempotency_key
        
        # Validate key format
        if not self._is_valid_key(idempotency_key):
            raise HTTPException(
                status_code=400,
                detail="Invalid X-Idempotency-Key format. Must be 16-128 characters."
            )
        
        # Check if already processed
        cache_key = f"idempotency:{idempotency_key}"
        cached_response = self.redis.get(cache_key)
        
        if cached_response:
            # Return cached response
            cached_data = json.loads(cached_response)
            return JSONResponse(
                content=cached_data["body"],
                status_code=cached_data["status_code"],
                headers={"X-Idempotency-Replay": "true"}
            )
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses (2xx status codes)
        if 200 <= response.status_code < 300:
            await self._cache_response(cache_key, response)
        
        return response
    
    def _should_skip_idempotency(self, request: Request) -> bool:
        """Check if endpoint should skip idempotency."""
        path = request.url.path
        
        # Skip webhooks (they have their own idempotency)
        if "/webhook" in path:
            return True
        
        # Skip health checks
        if path in ["/health", "/"]:
            return True
        
        return False
    
    def _is_valid_key(self, key: str) -> bool:
        """Validate idempotency key format."""
        if not key:
            return False
        
        # Key should be 16-128 characters
        if len(key) < 16 or len(key) > 128:
            return False
        
        # Key should be alphanumeric with some safe special chars
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        return all(c in allowed_chars for c in key)
    
    async def _cache_response(self, cache_key: str, response):
        """Cache response for idempotency replay."""
        # Read response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Parse as JSON
        try:
            body_json = json.loads(body)
        except json.JSONDecodeError:
            # Don't cache non-JSON responses
            return
        
        # Cache with TTL
        cache_data = {
            "body": body_json,
            "status_code": response.status_code,
            "cached_at": uuid.uuid4().hex  # For debugging
        }
        
        self.redis.setex(
            cache_key,
            self.key_ttl,
            json.dumps(cache_data)
        )


class IdempotencyKeyGenerator:
    """Helper class to generate idempotency keys."""
    
    @staticmethod
    def generate() -> str:
        """Generate a new idempotency key."""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_from_payload(payload: dict) -> str:
        """
        Generate deterministic idempotency key from payload.
        Useful for retries with same parameters.
        """
        import hashlib
        
        # Sort keys for consistency
        payload_str = json.dumps(payload, sort_keys=True)
        hash_obj = hashlib.sha256(payload_str.encode())
        return hash_obj.hexdigest()[:32]
