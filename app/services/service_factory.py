import httpx
import json
import asyncio
import base64
from typing import Dict, Any, Optional, Union
from uuid import UUID

from app.core.config import Config
from app.services.ai_service import OpenRouterService

class FalAIService:
    """Handles direct Fal.ai API"""
    
    def __init__(self, api_key: str):
        import fal_client
        fal_client.api_key = api_key
        self.fal_client = fal_client
    
    async def generate(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        # Remove 'fal-ai/' prefix if present
        if model_id.startswith("fal-ai/"):
            model_id = model_id[7:]
        
        handler = await asyncio.to_thread(
            self.fal_client.submit,
            model_id,
            arguments={
                "prompt": prompt,
                "image_size": kwargs.get('image_size', 'square_hd')
            }
        )
        
        result = await asyncio.to_thread(handler.get)
        
        if "images" in result:
            async with httpx.AsyncClient() as client:
                image_url = result["images"][0]["url"]
                image_resp = await client.get(image_url)
                return {
                    "type": "image",
                    "data": image_resp.content,
                    "format": "png"
                }
        
        return {"type": "raw", "data": result}

class GeminiService:
    """Handles direct Google Gemini API"""
    
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
    
    async def generate(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        model = self.genai.GenerativeModel(model_id)
        
        if kwargs.get('media_type') == 'image':
            response = await asyncio.to_thread(
                model.generate_images,
                prompt=prompt,
                number_of_images=1
            )
            import io
            img = response.images[0]
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return {
                "type": "image",
                "data": buf.getvalue(),
                "format": "png"
            }
        else:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={
                    "max_output_tokens": kwargs.get('max_tokens', 1000)
                }
            )
            return {
                "type": "text",
                "data": response.text
            }

class ServiceFactory:
    """Handles routing to different AI APIs based on service_type"""
    
    def __init__(self):
        self.services = {}
        self._init_services()
    
    def _init_services(self):
        """Initialize all API services"""
        # OpenRouter service
        if Config.OPENROUTER_API_KEY:
            self.services['openrouter'] = OpenRouterService()
        
        # Fal.ai service
        if Config.FAL_API_KEY:
            self.services['fal'] = FalAIService(
                api_key=Config.FAL_API_KEY
            )
        
        # Google Gemini service
        if Config.GEMINI_API_KEY:
            self.services['gemini'] = GeminiService(
                api_key=Config.GEMINI_API_KEY
            )
    
    async def generate(self, service_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route to appropriate service.
        Unified interface for all services.
        Payload should contain 'model_id', 'prompt', etc.
        """
        if service_type not in self.services:
             # Fallback to openrouter if not specified but available
             if 'openrouter' in self.services:
                 service_type = 'openrouter'
             else:
                raise ValueError(f"Service '{service_type}' not configured. Check API keys.")
        
        service = self.services[service_type]
        
        model_id = payload.get('model_id')
        prompt = payload.get('prompt')
        
        # Dispatch based on service type using their specific interfaces
        if service_type == 'openrouter':
            # OpenRouterService has generate_via_openrouter which returns bytes
            # We wrap it to return consistent Dict
            image_bytes = await service.generate_via_openrouter(prompt, model_id)
            return {
                "type": "image",
                "data": image_bytes,
                "format": "png" # Assumption
            }
        
        elif service_type == 'fal':
            return await service.generate(model_id, prompt, **payload)
            
        elif service_type == 'gemini':
            return await service.generate(model_id, prompt, **payload)
            
        else:
             raise ValueError(f"Unknown service type: {service_type}")

    async def close(self):
        """Clean up all services if needed"""
        pass
