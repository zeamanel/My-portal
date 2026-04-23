"""
Structured JSON Logging for Musa AI
Production-ready logging with correlation IDs.
"""

import json
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar

# Context variable for request correlation
request_id_var: ContextVar[str] = ContextVar('request_id', default='')


class StructuredLogger:
    """
    JSON structured logging for production observability.
    
    Usage:
        from app.core.logging import StructuredLogger as logger
        
        logger.info("Generation started", user_id="uuid", extra={"model_id": "flux-pro"})
        logger.error("Generation failed", user_id="uuid", extra={"error": str(e)})
    """
    
    SERVICE_NAME = "musa-ai-gateway"
    SERVICE_VERSION = "2.0.0"
    
    @staticmethod
    def _get_request_id() -> str:
        """Get current request ID from context."""
        return request_id_var.get() or str(uuid.uuid4())[:8]
    
    @classmethod
    def log(
        cls,
        level: str,
        message: str,
        user_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ):
        """
        Emit structured log entry.
        
        Args:
            level: Log level (DEBUG, INFO, WARN, ERROR, SECURITY)
            message: Log message
            user_id: Associated user ID
            extra: Additional structured data
            error: Exception to log (for ERROR level)
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.upper(),
            "message": message,
            "service": cls.SERVICE_NAME,
            "version": cls.SERVICE_VERSION,
            "request_id": cls._get_request_id()
        }
        
        if user_id:
            log_entry["user_id"] = user_id
        
        if extra:
            # Sanitize extra data (remove sensitive fields)
            sanitized = cls._sanitize_extra(extra)
            log_entry["extra"] = sanitized
        
        if error:
            log_entry["error"] = {
                "type": type(error).__name__,
                "message": str(error)
            }
        
        # Output to stderr for containerized environments
        print(json.dumps(log_entry), file=sys.stderr, flush=True)
    
    @classmethod
    def debug(cls, message: str, user_id: Optional[str] = None, **kwargs):
        """Debug level log."""
        cls.log("DEBUG", message, user_id, **kwargs)
    
    @classmethod
    def info(cls, message: str, user_id: Optional[str] = None, **kwargs):
        """Info level log."""
        cls.log("INFO", message, user_id, **kwargs)
    
    @classmethod
    def warn(cls, message: str, user_id: Optional[str] = None, **kwargs):
        """Warning level log."""
        cls.log("WARN", message, user_id, **kwargs)
    
    @classmethod
    def error(cls, message: str, user_id: Optional[str] = None, **kwargs):
        """Error level log."""
        cls.log("ERROR", message, user_id, **kwargs)
    
    @classmethod
    def security(cls, message: str, user_id: Optional[str] = None, **kwargs):
        """Security event log (for SIEM integration)."""
        cls.log("SECURITY", message, user_id, **kwargs)
    
    @staticmethod
    def _sanitize_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields from extra data."""
        sensitive_fields = [
            'password', 'token', 'secret', 'key', 'api_key',
            'private_key', 'credit_card', 'ssn', 'password'
        ]
        
        sanitized = {}
        for key, value in extra.items():
            # Check if key contains sensitive terms
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        
        return sanitized


class RequestIdMiddleware:
    """
    FastAPI middleware to set request ID for logging correlation.
    """
    
    async def __call__(self, request, call_next):
        # Get request ID from header or generate new one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        
        # Set in context
        token = request_id_var.set(request_id)
        
        # Add to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        # Reset context
        request_id_var.reset(token)
        
        return response
