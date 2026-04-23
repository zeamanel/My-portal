import os
import httpx
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

class APIServiceFactory:
    """Handles routing to different AI APIs based on service_type"""
    
    def __init__(self):
        self.services = {}
        self._init_services()
    
    def _init_services(self):
        """Initialize all API services"""
        # OpenRouter service
        self.services['openrouter'] = OpenRouterService(
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        # Fal.ai service
        if os.getenv("FAL_API_KEY"):
            self.services['fal'] = FalAIService(
                api_key=os.getenv("FAL_API_KEY")
            )
        
        # Google Gemini service
        if os.getenv("GEMINI_API_KEY"):
            self.services['gemini'] = GeminiService(
                api_key=os.getenv("GEMINI_API_KEY")
            )
        
        # Suno.ai service
        if os.getenv("SUNO_API_KEY"):
            self.services['suno'] = SunoService(
                api_key=os.getenv("SUNO_API_KEY")
            )
        
        # LaoZhang video service
        if os.getenv("LAOZHANG_API_KEY"):
            self.services['laozhang'] = LaoZhangService(
                api_key=os.getenv("LAOZHANG_API_KEY")
            )
        
        # Higgsfield service
        if os.getenv("HIGGSFIELD_API_KEY"):
            self.services['higgsfield'] = HiggsfieldService(
                api_key=os.getenv("HIGGSFIELD_API_KEY")
            )
    
    async def generate(self, service_type: str, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Route to appropriate service"""
        if service_type not in self.services:
            raise ValueError(f"Service '{service_type}' not configured. Check API keys.")
        
        service = self.services[service_type]
        return await service.generate(model_id, prompt, **kwargs)
    
    async def close(self):
        """Clean up all services"""
        for service in self.services.values():
            if hasattr(service, 'close'):
                await service.close()

class OpenRouterService:
    """Handles OpenRouter API calls for your 19 models"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://musa.ai",
            "X-Title": "Musa AI",
            "Content-Type": "application/json"
        }
        
        # Determine media type from model_id or kwargs
        media_type = kwargs.get('media_type', 'text')
        
        if media_type == 'image':
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "modalities": ["image"]
            }
        elif media_type == 'audio':
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "modalities": ["audio"]
            }
        else:  # text
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get('max_tokens', 1000)
            }
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter error: {response.text}")
        
        return await self._process_response(response.json(), media_type)
    
    async def _process_response(self, data: Dict, media_type: str) -> Dict[str, Any]:
        """Process OpenRouter response based on media type"""
        if media_type == 'image' and "choices" in data:
            choice = data["choices"][0]
            if "message" in choice and "images" in choice["message"]:
                image_data = choice["message"]["images"][0]
                image_url = image_data["image_url"]["url"]
                
                # Download image
                if image_url.startswith("data:image/"):
                    # Base64 image
                    header, base64_data = image_url.split(",", 1)
                    return {
                        "type": "image",
                        "data": base64.b64decode(base64_data),
                        "format": header.split("/")[1].split(";")[0]
                    }
                else:
                    # URL image
                    image_resp = await self.client.get(image_url)
                    return {
                        "type": "image",
                        "data": image_resp.content,
                        "format": "png"
                    }
        
        return {"type": "raw", "data": data}
    
    async def close(self):
        await self.client.aclose()

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
            import httpx
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

# Similar classes for SunoService, LaoZhangService, HiggsfieldService...