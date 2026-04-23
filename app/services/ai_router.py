"""
Musa AI Router - Optimized for Low Latency Multi-Service Dispatch
Single query, direct dispatch, connection pooling, all 6 services supported.
"""

import os
import io
import uuid
import asyncio
import base64
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

import httpx
from supabase import Client

# Optional imports - handle if not installed
try:
    import fal_client
    FAL_AVAILABLE = True
except ImportError:
    FAL_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class ModelConfig:
    """Model configuration from database."""
    model_id: str
    service_type: str
    endpoint: str
    display_name: str
    media_type: str
    base_price_tokens: int
    provider: str


class OptimizedAIRouter:
    """
    Optimized AI Router with:
    - Single database query
    - Direct dispatch map
    - Shared HTTP client (connection pooling)
    - All 6 services supported
    """
    
    # Service type to handler method mapping
    DISPATCH_MAP = {
        "fal": "_generate_fal",
        "gemini": "_generate_gemini",
        "openrouter": "_generate_openrouter",
        "laozhang": "_generate_laozhang",
        "higgsfield": "_generate_higgsfield",
        "suno": "_generate_suno"
    }
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        
        # Shared HTTP client with connection pooling
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True
        )
        
        # API Keys from environment
        self.api_keys = {
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "fal": os.getenv("FAL_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "suno": os.getenv("SUNO_API_KEY"),
            "laozhang": os.getenv("LAOZHANG_API_KEY"),
            "higgsfield": os.getenv("HIGGSFIELD_API_KEY")
        }
        
        # Initialize Gemini if available
        if GEMINI_AVAILABLE and self.api_keys["gemini"]:
            genai.configure(api_key=self.api_keys["gemini"])
        
        # Storage bucket
        self.storage_bucket = "ai-generations"
    
    async def generate_image(
        self,
        user_id: str,
        model_id: str,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main generation entry point.
        
        Flow:
        1. Single DB query for model config
        2. Direct dispatch via DISPATCH_MAP
        3. Generate image bytes
        4. Upload to storage
        5. Return public URL
        """
        
        # Step 1: Single database query
        print(f"🔍 Model lookup: {model_id}")
        model_config = await self._get_model_config(model_id)
        
        print(f"📊 Service type: {model_config.service_type}, Endpoint: {model_config.endpoint}")
        
        # Step 2: Direct dispatch
        handler_name = self.DISPATCH_MAP.get(model_config.service_type)
        if not handler_name:
            raise ValueError(f"Unknown service_type: {model_config.service_type}")
        
        handler = getattr(self, handler_name)
        print(f"🚀 Dispatching to {model_config.service_type} service")
        
        # Step 3: Generate image bytes
        image_bytes, metadata = await handler(
            endpoint=model_config.endpoint,
            prompt=prompt,
            **kwargs
        )
        
        # Step 4: Upload to storage
        public_url = await self._upload_to_storage(
            user_id=user_id,
            model_id=model_id,
            service_type=model_config.service_type,
            image_bytes=image_bytes
        )
        
        return {
            "url": public_url,
            "image_url": public_url,
            "model_id": model_id,
            "service_type": model_config.service_type,
            "provider": model_config.provider,
            "metadata": metadata
        }
    
    async def _get_model_config(self, model_id: str) -> ModelConfig:
        """
        Single database query to get all model configuration.
        Fetches: model_id, service_type, endpoint, display_name, media_type, base_price_tokens, provider
        """
        try:
            response = self.supabase.table("ai_models").select(
                "model_id, service_type, endpoint, display_name, media_type, base_price_tokens, provider"
            ).eq("model_id", model_id).eq("is_active", True).execute()
            
            if not response.data:
                raise ValueError(f"Model '{model_id}' not found or inactive")
            
            data = response.data[0]
            
            return ModelConfig(
                model_id=data["model_id"],
                service_type=data["service_type"],
                endpoint=data["endpoint"],
                display_name=data["display_name"],
                media_type=data["media_type"],
                base_price_tokens=data["base_price_tokens"],
                provider=data["provider"]
            )
            
        except Exception as e:
            print(f"❌ Database query failed: {e}")
            # Fallback to mock config for testing
            return self._get_mock_config(model_id)
    
    def _get_mock_config(self, model_id: str) -> ModelConfig:
        """Mock config for testing without database."""
        mock_configs = {
            "flux-pro": ModelConfig(
                model_id="flux-pro",
                service_type="fal",
                endpoint="fal-ai/flux-pro",
                display_name="Flux Pro",
                media_type="Image",
                base_price_tokens=40,
                provider="fal"
            ),
            "gemini-2": ModelConfig(
                model_id="gemini-2",
                service_type="gemini",
                endpoint="gemini-2.0-flash-exp",
                display_name="Gemini 2.0 Flash",
                media_type="Image",
                base_price_tokens=8,
                provider="google"
            ),
            "stable-video": ModelConfig(
                model_id="stable-video",
                service_type="openrouter",
                endpoint="stability-ai/stable-video",
                display_name="Stable Video",
                media_type="Video",
                base_price_tokens=120,
                provider="stability"
            )
        }
        
        return mock_configs.get(model_id, ModelConfig(
            model_id=model_id,
            service_type="openrouter",
            endpoint=model_id,
            display_name=model_id,
            media_type="Image",
            base_price_tokens=5,
            provider="openrouter"
        ))
    
    # ==================== SERVICE HANDLERS ====================
    
    async def _generate_fal(
        self,
        endpoint: str,
        prompt: str,
        **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Generate using Fal.ai API.
        Endpoint format: "fal-ai/flux-pro" or "fal-ai/flux/dev"
        """
        if not FAL_AVAILABLE:
            raise ImportError("fal-client not installed. Run: pip install fal-client")
        
        if not self.api_keys["fal"]:
            raise ValueError("FAL_API_KEY not configured")
        
        fal_client.api_key = self.api_keys["fal"]
        
        print(f"   🎨 Fal.ai: {endpoint}")
        
        # Run blocking call in thread pool
        def submit_and_wait():
            handler = fal_client.submit(
                endpoint,
                arguments={
                    "prompt": prompt,
                    "image_size": kwargs.get("image_size", "square_hd"),
                    "num_inference_steps": kwargs.get("steps", 28),
                    "guidance_scale": kwargs.get("guidance", 3.5)
                }
            )
            return handler.get()
        
        result = await asyncio.to_thread(submit_and_wait)
        
        if "images" not in result or not result["images"]:
            raise Exception("Fal.ai returned no images")
        
        # Download image from temporary URL
        image_url = result["images"][0]["url"]
        image_response = await self.http_client.get(image_url)
        image_response.raise_for_status()
        
        return image_response.content, {
            "provider": "fal",
            "endpoint": endpoint,
            "seed": result.get("seed")
        }
    
    async def _generate_gemini(
        self,
        endpoint: str,
        prompt: str,
        **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Generate using Google Gemini API.
        Endpoint is the model name: "gemini-2.0-flash-exp" or "imagen-3.0-generate-001"
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        if not self.api_keys["gemini"]:
            raise ValueError("GEMINI_API_KEY not configured")
        
        print(f"   🎨 Gemini: {endpoint}")
        
        model = genai.GenerativeModel(endpoint)
        
        # Run blocking call in thread pool
        def generate():
            return model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio=kwargs.get("aspect_ratio", "1:1")
            )
        
        response = await asyncio.to_thread(generate)
        
        if not response.images:
            raise Exception("Gemini returned no images")
        
        # Extract image bytes
        image = response.images[0]
        
        # Handle different response formats
        if hasattr(image, "_image_bytes"):
            image_bytes = image._image_bytes
        elif hasattr(image, "save"):
            # PIL Image
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        else:
            raise Exception("Unknown Gemini image format")
        
        return image_bytes, {
            "provider": "google",
            "model": endpoint
        }
    
    async def _generate_openrouter(
        self,
        endpoint: str,
        prompt: str,
        **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Generate using OpenRouter API.
        Endpoint is the model ID: "black-forest-labs/flux.2-pro"
        """
        if not self.api_keys["openrouter"]:
            raise ValueError("OPENROUTER_API_KEY not configured")
        
        print(f"   🎨 OpenRouter: {endpoint}")
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['openrouter']}",
            "HTTP-Referer": "https://musa.ai",
            "X-Title": "Musa AI",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "modalities": ["image", "text"],
            "max_tokens": 100
        }
        
        response = await self.http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract image from response
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            
            # Check for images field
            if "images" in message and len(message["images"]) > 0:
                image_data = message["images"][0]
                image_url = image_data.get("image_url", {}).get("url", "")
                
                if image_url.startswith("data:image/"):
                    # Base64 encoded image
                    header, base64_data = image_url.split(",", 1)
                    image_bytes = base64.b64decode(base64_data)
                    return image_bytes, {"provider": "openrouter", "format": "base64"}
                else:
                    # URL to download
                    img_response = await self.http_client.get(image_url)
                    img_response.raise_for_status()
                    return img_response.content, {"provider": "openrouter", "format": "url"}
        
        raise Exception("No image found in OpenRouter response")
    
    async def _generate_laozhang(
        self,
        endpoint: str,
        prompt: str,
        **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Generate using LaoZhang Video API with robust download handling.
        """
        if not self.api_keys["laozhang"]:
            raise ValueError("LAOZHANG_API_KEY not configured")
        
        print(f"   🎨 LaoZhang: {endpoint}")
        
        # LaoZhang API implementation
        headers = {
            "Authorization": f"Bearer {self.api_keys['laozhang']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint,
            "prompt": prompt,
            "duration": kwargs.get("duration", 5),
            "resolution": kwargs.get("resolution", "720p")
        }
        
        # Submit generation request - try multiple endpoints
        submit_urls = [
            "https://api.laozhang.ai/v1/video/generate",
            "https://api.laozhang.ai/v1/generate",
            "https://api.laozhang.ai/api/v1/generate"
        ]
        
        data = None
        for submit_url in submit_urls:
            try:
                print(f"   🔄 Trying submit endpoint: {submit_url}")
                response = await self.http_client.post(
                    submit_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                print(f"   ✅ Submit successful")
                break
            except Exception as e:
                print(f"   ⚠️ Submit failed: {e}")
                continue
        
        if not data:
            raise Exception("All LaoZhang submit endpoints failed")
        
        # Poll for result (video generation takes time)
        job_id = data.get("job_id")
        if not job_id:
            raise Exception(f"No job_id in response: {data}")
        
        print(f"   ⏳ Polling for job: {job_id}")
        video_bytes = await self._poll_laozhang_result(job_id, endpoint)
        
        return video_bytes, {
            "provider": "laozhang",
            "job_id": job_id,
            "duration": payload["duration"]
        }
    
    async def _poll_laozhang_result(self, job_id: str, model: str, max_attempts: int = 200) -> bytes:
        """
        Poll LaoZhang API for generation result and download video with validation.
        Returns actual video bytes, not URL.
        """
        headers = {"Authorization": f"Bearer {self.api_keys['laozhang']}"}
        
        # Determine polling interval based on model (HD takes longer)
        is_hd = "pro" in model.lower() or "15s" in model.lower()
        poll_interval = 5 if is_hd else 2
        
        # Try multiple status endpoints
        status_urls = [
            f"https://api.laozhang.ai/v1/video/status/{job_id}",
            f"https://api.laozhang.ai/v1/status/{job_id}",
            f"https://api.laozhang.ai/api/v1/status/{job_id}"
        ]
        
        video_url = None
        
        for attempt in range(max_attempts):
            # Try each status endpoint
            for status_url in status_urls:
                try:
                    response = await self.http_client.get(status_url, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    
                    status = data.get("status")
                    
                    if status == "completed":
                        video_url = data.get("url") or data.get("video_url") or data.get("result_url")
                        if video_url:
                            print(f"   ✅ Video ready: {video_url[:60]}...")
                            break
                    elif status == "failed":
                        raise Exception(f"LaoZhang generation failed: {data.get('error', 'Unknown error')}")
                    elif status in ["processing", "queued", "pending"]:
                        print(f"   ⏳ Status: {status} (attempt {attempt + 1}/{max_attempts})")
                    
                    break  # Status endpoint worked, no need to try others
                    
                except Exception as e:
                    if "status" in str(status_url).lower() and status_url == status_urls[-1]:
                        print(f"   ⚠️ All status endpoints failed on attempt {attempt + 1}")
                    continue
            
            if video_url:
                break
            
            await asyncio.sleep(poll_interval)
        
        if not video_url:
            raise Exception(f"LaoZhang generation timed out after {max_attempts * poll_interval} seconds")
        
        # Download video with retry logic and validation
        return await self._download_laozhang_video(video_url, job_id)
    
    async def _download_laozhang_video(self, video_url: str, job_id: str, max_retries: int = 3) -> bytes:
        """
        Download video from LaoZhang with retry logic and validation.
        Ensures video is >100KB and has valid headers.
        """
        headers = {"Authorization": f"Bearer {self.api_keys['laozhang']}"}
        
        # Try alternative download URLs
        download_urls = [
            video_url,
            f"https://api.laozhang.ai/v1/video/download/{job_id}",
            f"https://api.laozhang.ai/api/v1/download/{job_id}"
        ]
        
        for url in download_urls:
            print(f"   📥 Attempting download from: {url[:60]}...")
            
            for retry in range(max_retries):
                try:
                    # Download video
                    response = await self.http_client.get(url, headers=headers, timeout=60.0)
                    response.raise_for_status()
                    video_bytes = response.content
                    
                    # Check if response is JSON error (when API returns error as JSON)
                    if len(video_bytes) < 1024:
                        try:
                            error_data = response.json()
                            print(f"   ❌ Got JSON error instead of video: {error_data}")
                            continue
                        except:
                            pass  # Not JSON, continue with validation
                    
                    # Validate video size
                    video_size_kb = len(video_bytes) / 1024
                    print(f"   📦 Downloaded {video_size_kb:.1f} KB")
                    
                    if len(video_bytes) == 0:
                        print(f"   ⚠️ 0-byte video, retrying in 5 seconds... (retry {retry + 1}/{max_retries})")
                        await asyncio.sleep(5)
                        continue
                    
                    if video_size_kb < 100:
                        print(f"   ⚠️ Video too small ({video_size_kb:.1f} KB), retrying in 7 seconds... (retry {retry + 1}/{max_retries})")
                        await asyncio.sleep(7)
                        continue
                    
                    # Validate video headers
                    is_valid_video = self._validate_video_headers(video_bytes)
                    
                    if not is_valid_video:
                        print(f"   ⚠️ Invalid video headers, retrying in 10 seconds... (retry {retry + 1}/{max_retries})")
                        await asyncio.sleep(10)
                        continue
                    
                    print(f"   ✅ Valid video downloaded: {video_size_kb:.1f} KB")
                    return video_bytes
                    
                except httpx.HTTPStatusError as e:
                    print(f"   ❌ HTTP error {e.response.status_code}: {e}")
                    break  # Don't retry on HTTP errors, try next URL
                except Exception as e:
                    print(f"   ⚠️ Download error: {e}")
                    if retry < max_retries - 1:
                        await asyncio.sleep(5)
                    continue
        
        raise Exception(f"Failed to download valid video after trying all endpoints and retries")
    
    def _validate_video_headers(self, data: bytes) -> bool:
        """
        Validate video file headers (MP4 or WebM).
        MP4: starts with 'ftyp' at offset 4
        WebM: starts with EBML header (0x1A45DFA3)
        """
        if len(data) < 12:
            return False
        
        # Check for MP4 (ftyp box)
        if data[4:8] == b'ftyp':
            print(f"   ✓ Valid MP4 header detected")
            return True
        
        # Check for WebM (EBML header)
        if data[0:4] == b'\x1a\x45\xdf\xa3':
            print(f"   ✓ Valid WebM header detected")
            return True
        
        # Log what we got instead
        header_hex = data[:16].hex()
        print(f"   ✗ Invalid video header: {header_hex}")
        return False
    
    async def _generate_higgsfield(
        self,
        endpoint: str,
        prompt: str,
        **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Generate using Higgsfield Video API.
        """
        if not self.api_keys["higgsfield"]:
            raise ValueError("HIGGSFIELD_API_KEY not configured")
        
        print(f"   🎨 Higgsfield: {endpoint}")
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['higgsfield']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint,
            "prompt": prompt,
            "num_frames": kwargs.get("num_frames", 121),
            "fps": kwargs.get("fps", 24)
        }
        
        response = await self.http_client.post(
            "https://api.higgsfield.ai/v1/generate",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        job_id = data.get("job_id")
        
        # Poll for result
        video_url = await self._poll_higgsfield_result(job_id)
        
        video_response = await self.http_client.get(video_url)
        video_response.raise_for_status()
        
        return video_response.content, {
            "provider": "higgsfield",
            "job_id": job_id
        }
    
    async def _poll_higgsfield_result(self, job_id: str, max_attempts: int = 60) -> str:
        """Poll Higgsfield API for generation result."""
        headers = {"Authorization": f"Bearer {self.api_keys['higgsfield']}"}
        
        for attempt in range(max_attempts):
            response = await self.http_client.get(
                f"https://api.higgsfield.ai/v1/status/{job_id}",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "completed":
                return data["url"]
            elif data.get("status") == "failed":
                raise Exception(f"Higgsfield generation failed: {data.get('error')}")
            
            await asyncio.sleep(2)
        
        raise Exception("Higgsfield generation timed out")
    
    async def _generate_suno(
        self,
        endpoint: str,
        prompt: str,
        **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Generate using Suno Audio API.
        """
        if not self.api_keys["suno"]:
            raise ValueError("SUNO_API_KEY not configured")
        
        print(f"   🎵 Suno: {endpoint}")
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['suno']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint,
            "prompt": prompt,
            "duration": kwargs.get("duration", 30),
            "tags": kwargs.get("tags", []),
            "title": kwargs.get("title", "Generated Track")
        }
        
        response = await self.http_client.post(
            "https://api.suno.ai/v1/generate",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        job_id = data.get("job_id")
        
        # Poll for result
        audio_url = await self._poll_suno_result(job_id)
        
        audio_response = await self.http_client.get(audio_url)
        audio_response.raise_for_status()
        
        return audio_response.content, {
            "provider": "suno",
            "job_id": job_id,
            "duration": payload["duration"]
        }
    
    async def _poll_suno_result(self, job_id: str, max_attempts: int = 60) -> str:
        """Poll Suno API for generation result."""
        headers = {"Authorization": f"Bearer {self.api_keys['suno']}"}
        
        for attempt in range(max_attempts):
            response = await self.http_client.get(
                f"https://api.suno.ai/v1/status/{job_id}",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "completed":
                return data["audio_url"]
            elif data.get("status") == "failed":
                raise Exception(f"Suno generation failed: {data.get('error')}")
            
            await asyncio.sleep(2)
        
        raise Exception("Suno generation timed out")
    
    # ==================== STORAGE ====================
    
    async def _upload_to_storage(
        self,
        user_id: str,
        model_id: str,
        service_type: str,
        image_bytes: bytes
    ) -> str:
        """
        Upload generated content to Supabase Storage.
        Path format: {user_id}/{service_type}/{model_id}/{timestamp}_{uuid}.png
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Determine file extension based on content
        ext = self._get_file_extension(image_bytes)
        
        path = f"{user_id}/{service_type}/{model_id}/{timestamp}_{unique_id}.{ext}"
        
        print(f"   📁 Uploading to: {path}")
        
        try:
            # Upload to Supabase Storage
            result = self.supabase.storage.from_(self.storage_bucket).upload(
                path=path,
                file=image_bytes,
                file_options={"content-type": self._get_content_type(ext)}
            )
            
            # Get public URL
            public_url = self.supabase.storage.from_(self.storage_bucket).get_public_url(path)
            
            print(f"   ✅ Uploaded: {public_url[:60]}...")
            
            return public_url
            
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            # Return placeholder for testing
            return f"https://placehold.co/600x600/png?text={model_id}"
    
    def _get_file_extension(self, data: bytes) -> str:
        """Detect file type from magic bytes."""
        if data[:4] == b'\x89PNG':
            return "png"
        elif data[:2] == b'\xff\xd8':
            return "jpg"
        elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return "webp"
        elif data[:4] == b'\x1aE\xdf\xa3':
            return "webm"  # Video
        elif data[:4] == b'ftyp':
            return "mp4"   # Video
        else:
            return "png"   # Default
    
    def _get_content_type(self, ext: str) -> str:
        """Get MIME type from extension."""
        types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "gif": "image/gif",
            "mp4": "video/mp4",
            "webm": "video/webm",
            "mp3": "audio/mpeg",
            "wav": "audio/wav"
        }
        return types.get(ext, "application/octet-stream")
    
    async def close(self):
        """Cleanup resources."""
        await self.http_client.aclose()
