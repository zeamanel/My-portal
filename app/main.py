# -*- coding: utf-8 -*-
"""
Musa AI Gateway - Final Production Version
Integrated with robust OpenRouter logic and Video Storage Support
"""

import os
import uuid
import asyncio
import base64
import re
import httpx
import time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from app.services.chapa_integration import ChapaIntegration
from cachetools import TTLCache  # you may need to install cachetools
from fastapi.responses import HTMLResponse

# Load environment variables
load_dotenv()

# ==================== SUPABASE SETUP ====================
try:
    from supabase import create_client, Client
except ImportError:
    raise ImportError("supabase not installed. Run: pip install supabase")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")

# Initialize Client
base_url = supabase_url.strip().rstrip('/') + '/'
supabase: Client = create_client(base_url, supabase_key.strip())
print(f"✅ Supabase connected: {base_url[:30]}...")

# ==================== HTTP CLIENT ====================
try:
    import httpx
except ImportError:
    raise ImportError("httpx not installed. Run: pip install httpx")

# ==================== OPTIONAL SDKS ====================
try:
    import fal_client
    FAL_AVAILABLE = True
except ImportError:
    FAL_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==================== PYDANTIC MODELS ====================

# Define ElementInput first so GenerationRequest can use it
class ElementInput(BaseModel):
    reference_image_urls: List[str]
    frontal_image_url: str

class GenerationRequest(BaseModel):
    user_id: str
    prompt: str
    model_id: str = "auto"
    template_id: Optional[str] = None
    quality: str = "1K"
    aspect_ratio: str = "1:1"
    is_public: bool = True
    image_url: Optional[str] = None
    context_toggle: bool = False
    country_code: Optional[str] = None
    # New video-to-video fields
    video_url: Optional[str] = None
    elements: Optional[List[ElementInput]] = None
    duration: Optional[str] = None          # fal.ai expects string like "5" or "10"
    keep_audio: Optional[bool] = None
    skip_enhancement: bool = False
    context_images: Optional[List[str]] = None   # <-- NEW
 # New music‑specific fields
    music_custom_mode: Optional[bool] = False
    music_style: Optional[str] = None
    music_title: Optional[str] = None
    music_lyrics: Optional[str] = None          # if custom mode, use this as lyrics; else use prompt
    music_num_tracks: Optional[int] = 1         # 1 or 2 (Suno returns two tracks when possible)

class ChapaInitiateRequest(BaseModel):
    user_id: str
    amount_etb: int
    phone_number: str
    callback_url: str = "http://localhost:8000/api/v1/chapa/verify"

# ==================== OPTIMIZED AI ROUTER ====================

class OptimizedAIRouter:
    DISPATCH_MAP = {
        "fal": "_generate_fal",
        "gemini": "_generate_gemini",
        "openrouter": "_generate_openrouter",
        "video_generation": "_generate_laozhang",
        "laozhang": "_generate_laozhang",
        "higgsfield": "_generate_higgsfield",
        "suno": "_generate_suno"
    }
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.context_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour cache
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            http2=True
        )
        self.api_keys = {
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "fal": os.getenv("FAL_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "laozhang": os.getenv("LAOZHANG_API_KEY"),
            "suno": os.getenv("SUNO_API_KEY"),
        }
        self.storage_bucket = "ai-generations"

    async def _enhance_prompt(self, original_prompt: str, model_id: str, context_descriptions: Optional[List[str]] = None, media_type: str = "Image") -> str:
        if not GEMINI_AVAILABLE:
            return original_prompt

        context_text = ""
        if context_descriptions:
            bullets = "\n".join(f"- {desc}" for desc in context_descriptions)
            context_text = f"\nRelevant cultural reference images have these descriptions:\n{bullets}\nIncorporate elements from these descriptions into the prompt naturally."

        if media_type == "Audio":
            enhancement_prompt = f"""
You are an expert music prompt engineer. Rewrite the following user request into a detailed music generation prompt suitable for Suno AI.{context_text}

Original user request: "{original_prompt}"

The enhanced prompt should:
- Include a clear genre/style (e.g., "lo‑fi hip hop", "orchestral", "electronic pop").
- Suggest tempo, mood, instrumentation, or vocal style if implied.
- Be concise but evocative.
- Return ONLY the enhanced prompt, no extra text.

Enhanced prompt:
"""
        else:
            enhancement_prompt = f"""
You are an expert prompt engineer. Rewrite the following prompt to be more effective for the AI model "{model_id}".{context_text}

Original prompt: "{original_prompt}"

Requirements:
- Keep the core idea intact.
- Add relevant details (style, lighting, composition, mood) that suit the model's strengths.
- Be clear, descriptive, and concise.
- Return ONLY the enhanced prompt, no explanations.

Enhanced prompt:
"""

        try:
            client = genai.Client(api_key=self.api_keys["gemini"])
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=enhancement_prompt,
                )
            )
            enhanced = response.text.strip()
            if enhanced:
                print(f"   ✨ Enhanced {'music' if media_type == 'Audio' else 'prompt'}: {enhanced[:80]}...")
                return enhanced
        except Exception as e:
            print(f"   ⚠️ Prompt enhancement failed: {e}")

        return original_prompt

    async def generate_image(self, user_id: str, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        model_config = await self._get_model_config(model_id)
        media_type = model_config.get("media_type", "Image")
        kwargs["media_type"] = media_type
        service_type = model_config['service_type']

        # --- Context image retrieval ---
        # Existing context_images from frontend (e.g., template images)
        existing_context_images = kwargs.get("context_images", [])
        context_images_db = []
        context_descriptions_db = []

        if kwargs.get("country_code"):
            cache_key = f"country:{kwargs['country_code']}"
            cached = self.context_cache.get(cache_key) if hasattr(self, 'context_cache') else None
            if cached:
                context_images_db, context_descriptions_db = cached
                print(f"   📸 Using cached context images for country {kwargs['country_code']} ({len(context_images_db)} found)")
            else:
                try:
                    res = self.supabase.table("context_images") \
                        .select("image_url, description") \
                        .eq("country_code", kwargs["country_code"]) \
                        .limit(10) \
                        .execute()
                    if res.data:
                        context_images_db = [item["image_url"] for item in res.data]
                        context_descriptions_db = [item["description"] for item in res.data if item.get("description")]
                        if hasattr(self, 'context_cache'):
                            self.context_cache[cache_key] = (context_images_db, context_descriptions_db)
                        print(f"   📸 Fetched {len(context_images_db)} context images for country {kwargs['country_code']}")
                except Exception as e:
                    print(f"   ⚠️ Failed to fetch context images: {e}")

        # Merge frontend and database images (frontend first)
        all_context_images = existing_context_images + context_images_db
        if all_context_images:
            kwargs["context_images"] = all_context_images
            # Only database images have descriptions; use them for enhancer
            kwargs["context_descriptions"] = context_descriptions_db
            print(f"   🖼️ Total context images: {len(all_context_images)} (frontend: {len(existing_context_images)}, DB: {len(context_images_db)})")
        # --- End context retrieval ---

        # Pass model_id to handlers (like Suno) that need it
        kwargs["model_id"] = model_id

        # Optional enhancement step (per-model flag AND per-request override)
        enhance = model_config.get("enhance_prompt", True)  # default True
        skip = kwargs.pop("skip_enhancement", False)        # consume the flag so it isn't forwarded to handlers
        if enhance and not skip:
            prompt = await self._enhance_prompt(prompt, model_id, context_descriptions=context_descriptions_db, media_type=media_type)
        elif skip:
            print("   ⏭️ Prompt enhancement skipped (skip_enhancement=True)")

        handler_name = self.DISPATCH_MAP.get(service_type)
        if not handler_name:
            raise ValueError(f"Unknown service_type: {service_type}")

        handler = getattr(self, handler_name)
        endpoint = model_config.get('endpoint') or model_id

        content_bytes, metadata = await handler(endpoint=endpoint, prompt=prompt, **kwargs)

        public_url = await self._upload_to_storage(
            user_id=user_id, model_id=model_id, service_type=service_type, content_bytes=content_bytes
        )

        return {
            "url": public_url,
            "image_url": public_url,
            "model_id": model_id,
            "provider": model_config.get('provider'),
            "metadata": metadata
        }

    async def _get_model_config(self, model_id: str) -> Dict[str, Any]:
        response = self.supabase.table("ai_models").select("*").eq("model_id", model_id).execute()
        if not response.data:
            raise ValueError(f"Model '{model_id}' not found")
        return response.data[0]

    async def _generate_openrouter(self, endpoint: str, prompt: str, **kwargs) -> tuple:
        """
        Generate using OpenRouter API – supports multiple images.
        """
        if not self.api_keys["openrouter"]:
            raise ValueError("OPENROUTER_API_KEY not configured")

        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_keys['openrouter']}",
            "HTTP-Referer": "https://musa.ai",
            "X-Title": "Musa AI",
            "Content-Type": "application/json"
        }

        # Build content array
        content = [{"type": "text", "text": prompt}]
        user_image = kwargs.get("image_url") or kwargs.get("start_image_url")
        if user_image:
            content.append({"type": "image_url", "image_url": {"url": user_image.strip()}})
        for ctx_url in kwargs.get("context_images", []):
            content.append({"type": "image_url", "image_url": {"url": ctx_url}})

        messages = [{"role": "user", "content": content}]

        payload = {
            "model": endpoint,
            "messages": messages
        }

        # Add model-specific extras
        is_gemini = "gemini" in endpoint.lower()
        is_flux = "flux" in endpoint.lower()

        if is_gemini:
            payload["modalities"] = ["image", "text"]
            payload["image_config"] = {"aspect_ratio": "1:1", "image_size": "1K"}
        elif is_flux:
            payload["modalities"] = ["image"]
        else:
            payload["max_tokens"] = 500

        try:
            response = await self.http_client.post(api_url, headers=headers, json=payload)
            if response.status_code != 200:
                error_msg = response.json().get('error', {}).get('message', response.text[:200])
                raise ValueError(f"OpenRouter Error: {error_msg}")

            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                if "images" in message and len(message["images"]) > 0:
                    img_entry = message["images"][0]
                    image_url = img_entry.get("image_url", {}).get("url") or img_entry.get("url", "")

                    if image_url.startswith("data:image/"):
                        header, base64_data = image_url.split(",", 1)
                        return base64.b64decode(base64_data), {"provider": "openrouter", "format": "base64"}
                    else:
                        img_res = await self.http_client.get(image_url)
                        return img_res.content, {"provider": "openrouter", "format": "url"}

            raise Exception("No image found in OpenRouter response")
        except Exception as e:
            raise ValueError(f"OpenRouter failed: {str(e)}")

    async def _generate_laozhang(self, endpoint: str, prompt: str, **kwargs) -> tuple:
        """
        Generate video using LaoZhang Veo API (OpenAI-compatible streaming).
        Model endpoint examples: "veo-3.1-fast", "veo-3.1-generate-preview"
        """
        import re
        import json

        api_key = self.api_keys.get("laozhang")
        if not api_key:
            raise ValueError("LAOZHANG_API_KEY not configured")

        base_url = "https://api.laozhang.ai/v1"
        url = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Payload exactly as in the successful test
        payload = {
            "model": endpoint,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "stream": True,
            "n": 1
        }

        print(f"   🎬 LaoZhang Veo request: {url}")
        print(f"   📦 Model: {endpoint}")
        print(f"   📝 Prompt: {prompt[:80]}...")

        # Use a longer timeout for video generation (up to 5 minutes)
        timeout = httpx.Timeout(300.0, connect=10.0)

        video_url = None
        full_response_text = ""

        try:
            # Stream the response to get progress and final URL
            async with self.http_client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as response:
                response.raise_for_status()

                print(f"   ⏳ Streaming response...")
                async for line in response.aiter_lines():
                    if line.strip():
                        print(f"   {line[:100]}")  # Log first 100 chars
                        full_response_text += line + "\n"

                        # Look for the success line containing the video URL
                        # Pattern: ✅ 视频生成成功，[点击这里](https://...mp4) 查看视频~~~
                        match = re.search(r'https?://[^\s]+\.mp4', line)
                        if match:
                            video_url = match.group(0)
                            print(f"   ✅ Found video URL: {video_url}")
                            # Don't break; let the stream finish for logging
                print(f"   ✅ Streaming complete")

            if not video_url:
                raise ValueError(f"No video URL found in response. Full response: {full_response_text[:500]}")

            # Download the video from the obtained URL
            print(f"   ⬇️ Downloading video from {video_url}...")
            video_response = await self.http_client.get(video_url, timeout=60.0)
            video_response.raise_for_status()
            video_bytes = video_response.content

            print(f"   ✅ Video downloaded: {len(video_bytes)} bytes")

            return video_bytes, {
                "provider": "laozhang",
                "model": endpoint,
                "video_url_original": video_url,
                "size_bytes": len(video_bytes)
            }

        except Exception as e:
            print(f"   ❌ LaoZhang generation error: {e}")
            raise ValueError(f"LaoZhang failed: {str(e)}")

    async def _generate_gemini(self, endpoint: str, prompt: str, **kwargs) -> tuple:
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai missing")

        client = genai.Client(api_key=self.api_keys["gemini"])
        model_id = endpoint.replace("models/", "").split(":")[0]

        # Collect all image URLs: user's reference + context images
        all_image_urls = []
        user_image = kwargs.get("image_url") or kwargs.get("start_image_url")
        if user_image:
            all_image_urls.append(user_image.strip())
        context_images = kwargs.get("context_images", [])
        all_image_urls.extend(context_images)

        # Build content parts
        parts = [types.Part.from_text(text=prompt)]
        for url in all_image_urls:
            try:
                img_resp = await self.http_client.get(url, timeout=10.0)
                img_resp.raise_for_status()
                img_bytes = img_resp.content
                mime = img_resp.headers.get("content-type", "image/jpeg")
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
                print(f"   ✅ Added image from {url[:60]}...")
            except Exception as e:
                print(f"   ⚠️ Could not download {url}: {e}")

        print(f"   🖼️ Total images sent: {len(parts)-1}")
        if len(parts) == 1:
            contents = prompt  # text only
        else:
            contents = parts

        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
            )
        )

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return part.inline_data.data, {"provider": "google-native"}
        raise Exception("Gemini native generation failed")

    async def _generate_fal(self, endpoint: str, prompt: str, **kwargs) -> tuple:
        """
        Generate using fal.ai API (via official client) for both images and videos.
        """
        if not FAL_AVAILABLE:
            raise ImportError("fal-client not installed. Run: pip install fal-client")
        if not self.api_keys["fal"]:
            raise ValueError("FAL_API_KEY not configured")

        fal_client.api_key = self.api_keys["fal"]

        # Build arguments dictionary from prompt and any additional kwargs
        arguments = {"prompt": prompt}
        # Add optional parameters that fal.ai models accept
        for key in ["image_url", "start_image_url", "duration", "aspect_ratio", "generate_audio",
                    "source_image_url", "target_video_url", "audio_url", "negative_prompt", "cfg_scale",
                    "video_url", "elements", "keep_audio"]:
            if key in kwargs:
                arguments[key] = kwargs[key]

        print(f"   🎬 fal.ai request: endpoint={endpoint}")
        print(f"   📦 Arguments keys: {list(arguments.keys())}")

        try:
            # Run the blocking subscribe call in a thread
            result = await asyncio.to_thread(
                fal_client.subscribe,
                endpoint,
                arguments=arguments,
            )

            # Extract media URL – handle different response structures
            media_url = None
            if "video" in result and isinstance(result["video"], dict) and "url" in result["video"]:
                media_url = result["video"]["url"]
            elif "images" in result and len(result["images"]) > 0 and "url" in result["images"][0]:
                media_url = result["images"][0]["url"]
            elif "image" in result and isinstance(result["image"], dict) and "url" in result["image"]:
                media_url = result["image"]["url"]
            elif "url" in result:
                media_url = result["url"]

            if not media_url:
                raise Exception(f"No media URL found in fal.ai response: {result}")

            print(f"   ⬇️ Downloading from {media_url[:100]}...")
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    media_response = await self.http_client.get(media_url, timeout=60.0)
                    media_response.raise_for_status()
                    media_bytes = media_response.content
                    break
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise  # re-raise the last error
                    print(f"   ⚠️ Download attempt {attempt+1} failed: {e}. Retrying in {2**attempt}s...")
                    await asyncio.sleep(2**attempt)  # exponential backoff

            # Determine media type from content-type header
            content_type = media_response.headers.get("content-type", "")
            media_type = "video" if "video" in content_type else "image"

            return media_bytes, {
                "provider": "fal",
                "model": endpoint,
                "media_type": media_type,
                "size_bytes": len(media_bytes),
                "original_url": media_url,
                "seed": result.get("seed")
            }

        except Exception as e:
            print(f"   ❌ fal.ai generation error: {e}")
            raise ValueError(f"fal.ai generation failed: {str(e)}")

    async def _generate_higgsfield(self, **k):
        return b"", {}

    async def _generate_suno(self, endpoint: str, prompt: str, **kwargs) -> tuple:
        api_key = self.api_keys.get("suno")
        if not api_key:
            raise ValueError("SUNO_API_KEY not configured")

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        model = kwargs.get("model_id", "V4_5")
        public_url = os.getenv("PUBLIC_URL", "https://musa-ai-backend-760742977917.us-central1.run.app")
        callback_url = f"{public_url}/api/v1/suno/callback"

        custom_mode = kwargs.get("custom_mode", False)

        # ✅ NEW: Support audio_url for continuation / extend (with lyrics in custom mode)
        audio_url = kwargs.get("audio_url")   # previous song's audioUrl (from earlier Suno generation)

        submit_payload = {
            "model": model,
            "callBackUrl": callback_url,
            "customMode": custom_mode,
            "instrumental": False,
        }

        # Add audio continuation if provided
        if audio_url:
            submit_payload["audioUrl"] = audio_url
            print(f"   🔄 Continuation mode enabled – using previous audio: {audio_url[:80]}...")

        if custom_mode:
            submit_payload["style"] = kwargs.get("style", "pop")
            submit_payload["title"] = kwargs.get("title", "Generated Track")
            submit_payload["prompt"] = kwargs.get("lyrics", prompt)
            if kwargs.get("num_tracks") == 2:
                submit_payload["num_tracks"] = 2
        else:
            submit_payload["prompt"] = prompt

        print(f"   🎵 Suno generation request: {endpoint} | Custom: {custom_mode} | Continuation: {bool(audio_url)}")
        print(f"   📦 Full submit payload: {submit_payload}")

        # Submit
        submit_response = await self.http_client.post(endpoint, json=submit_payload, headers=headers)
        submit_response.raise_for_status()
        submit_data = submit_response.json()

        task_id = submit_data.get("data", {}).get("taskId") or submit_data.get("taskId")
        if not task_id:
            raise ValueError(f"No taskId found in submit response: {submit_data}")

        # Polling
        status_url = "https://api.sunoapi.org/api/v1/generate/record-info"
        for attempt in range(120):
            await asyncio.sleep(5)
            try:
                resp = await self.http_client.get(f"{status_url}?taskId={task_id}", headers=headers)
                if resp.status_code != 200:
                    continue

                status_data = resp.json()
                inner = status_data.get("data", {})
                state = inner.get("status", "").upper()

                if state == "SUCCESS":
                    response_obj = inner.get("response", {})
                    suno_data = response_obj.get("sunoData", [])

                    if not suno_data:
                        raise Exception("No sunoData returned")

                    track = suno_data[0]  # first track
                    audio_url_final = track.get("audioUrl")
                    image_url = track.get("imageUrl")   # cover art

                    if not audio_url_final:
                        raise Exception("No audioUrl found")

                    # Download audio
                    audio_resp = await self.http_client.get(audio_url_final, timeout=60.0)
                    audio_resp.raise_for_status()
                    audio_bytes = audio_resp.content

                    # Download cover image (if available)
                    cover_bytes = None
                    cover_metadata = {}
                    if image_url:
                        try:
                            img_resp = await self.http_client.get(image_url, timeout=30.0)
                            img_resp.raise_for_status()
                            cover_bytes = img_resp.content
                            print(f"   🖼️ Cover image downloaded: {len(cover_bytes)} bytes")
                            cover_metadata = {"cover_url": image_url, "cover_size": len(cover_bytes)}
                        except Exception as img_err:
                            print(f"   ⚠️ Failed to download cover image: {img_err}")

                    print(f"   ✅ Suno generation completed! Audio: {len(audio_bytes)} bytes")

                    metadata = {
                        "provider": "suno",
                        "task_id": task_id,
                        "audio_url": audio_url_final,
                        **cover_metadata
                    }

                    # Return audio bytes + metadata (cover is now included)
                    return audio_bytes, metadata

                elif state in ("FAILED", "ERROR"):
                    raise Exception(f"Suno failed: {inner.get('msg')}")

            except Exception as e:
                print(f"   ⚠️ Poll error: {e}")
                continue

        raise TimeoutError("Suno generation timed out")

    async def _upload_to_storage(self, user_id, model_id, service_type, content_bytes):
        """
        Supports images (jpg), videos (mp4), and audio (mp3).
        """
        if service_type == "suno":
            ext = "mp3"
            mime = "audio/mpeg"
        elif service_type in ["video_generation", "laozhang", "fal"]:
            ext = "mp4"
            mime = "video/mp4"
        else:
            ext = "jpg"
            mime = "image/jpeg"

        filename = f"{uuid.uuid4()}.{ext}"
        path = f"{user_id}/{service_type}/{model_id}/{filename}".replace("//", "/")
        self.supabase.storage.from_(self.storage_bucket).upload(
            path=path,
            file=content_bytes,
            file_options={"content-type": mime, "x-upsert": "true"}
        )
        public_url = self.supabase.storage.from_(self.storage_bucket).get_public_url(path)
        print(f"   ✅ Uploaded to Supabase: {public_url}")
        return public_url

    async def close(self):
        await self.http_client.aclose()

# ==================== GATEKEEPER ====================

class Gatekeeper:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    async def validate_user(self, user_id: str):
        res = self.supabase.table("profiles").select("*").eq("id", user_id).execute()
        if not res.data: 
            raise HTTPException(404, "User not found")
        return res.data[0]

    async def process_transaction(self, user_id, cost, model_id):
        res = self.supabase.rpc("process_musa_transaction", {
            "p_user_id": user_id, 
            "p_total_cost": cost, 
            "p_model_id": model_id, 
            "p_base_cost": cost
        }).execute()
        
        if hasattr(res, 'data') and isinstance(res.data, dict) and "error_message" in res.data:
            raise HTTPException(status_code=400, detail=res.data["error_message"])
        return res.data

# ==================== APP LIFECYCLE ====================

ai_router = OptimizedAIRouter(supabase)
gatekeeper = Gatekeeper(supabase)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await ai_router.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/v1/models/gated")
async def get_models():
    res = supabase.table("ai_models").select("*").eq("is_active", True).execute()
    return {"status": "success", "models": res.data}

@app.post("/api/v1/generate")
async def generate(request: GenerationRequest):
    user = await gatekeeper.validate_user(request.user_id)
    
    model_res = supabase.table("ai_models").select("*").eq("model_id", request.model_id).execute()
    if not model_res.data:
        raise HTTPException(404, f"Model '{request.model_id}' not found")
    model = model_res.data[0]
    
    cost = model.get("base_price_tokens", 0)
    if user.get("credit_balance", 0) < cost:
        raise HTTPException(402, "Insufficient balance")
    
    tx = await gatekeeper.process_transaction(request.user_id, cost, request.model_id)
    
    try:
        # Build optional kwargs
        extra_kwargs: Dict[str, Any] = {}
        if request.image_url:
            extra_kwargs["image_url"] = request.image_url
        if request.context_toggle and request.country_code:
            extra_kwargs["country_code"] = request.country_code
        if request.context_images:
            extra_kwargs["context_images"] = request.context_images

        # Music‑specific fields
        extra_kwargs["custom_mode"] = request.music_custom_mode
        if request.music_style:
            extra_kwargs["style"] = request.music_style
        if request.music_title:
            extra_kwargs["title"] = request.music_title
        if request.music_lyrics:
            extra_kwargs["lyrics"] = request.music_lyrics
        if request.music_num_tracks:
            extra_kwargs["num_tracks"] = request.music_num_tracks

        # Media type from the model (so the enhancer knows it's audio)
        media_type = model.get("media_type", "Image")  # from the ai_models row
        extra_kwargs["media_type"] = media_type
        # Video-to-video fields
        if request.video_url:
            extra_kwargs["video_url"] = request.video_url
        if request.elements:
            # Convert list of Pydantic models to list of dicts
            extra_kwargs["elements"] = [el.dict() for el in request.elements]
        if request.duration:
            extra_kwargs["duration"] = request.duration
        if request.keep_audio is not None:
            extra_kwargs["keep_audio"] = request.keep_audio
        if request.skip_enhancement:
            extra_kwargs["skip_enhancement"] = True

        generation = await ai_router.generate_image(
            user_id=request.user_id, 
            model_id=request.model_id, 
            prompt=request.prompt,
            **extra_kwargs
        )
        return {**generation, "remaining_balance": tx.get("new_balance") if tx else 0}
    except Exception as e:
        print(f"Generation failed: {str(e)}")
        raise HTTPException(500, f"Generation failed: {str(e)}")

@app.options("/api/v1/generate")
async def options_generate():
    return {"detail": "OK"}
# ==================== TEMPLATE ENDPOINTS ====================

@app.get("/api/v1/templates")
async def get_templates(is_published: bool = True):
    """
    Retrieve all published templates.
    """
    try:
        res = supabase.table("creator_templates") \
            .select("*") \
            .eq("is_published", is_published) \
            .execute()
        return {"status": "success", "templates": res.data}
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch templates: {str(e)}")

@app.get("/api/v1/templates/{template_id}/fields")
async def get_template_fields(template_id: str):
    """
    Retrieve fields for a specific template.
    """
    try:
        res = supabase.table("template_fields") \
            .select("*") \
            .eq("template_id", template_id) \
            .order("display_order") \
            .execute()
        return {"status": "success", "fields": res.data}
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch template fields: {str(e)}")
# ==================== CHAPA PAYMENT ROUTES ====================

def _get_chapa() -> ChapaIntegration:
    """Lazy-initialize ChapaIntegration (avoids crash if webhook secret is a placeholder at startup)."""
    return ChapaIntegration(supabase)

@app.post("/api/v1/chapa/initiate")
async def chapa_initiate(request: ChapaInitiateRequest):
    chapa = _get_chapa()
    result = await chapa.initiate_payment(
        user_id=request.user_id,
        amount_etb=request.amount_etb,
        phone_number=request.phone_number,
        callback_url=request.callback_url,
    )
    return result

# Helper function to perform the actual verification with Chapa API
async def _verify_with_chapa(tx_ref: str) -> Dict[str, Any]:
    chapa = _get_chapa()
    secret_key = chapa.secret_key
    if not secret_key:
        raise HTTPException(500, "Chapa secret key not configured")

    async with httpx.AsyncClient(timeout=30.0) as client:
        url = f"{chapa.CHAPA_API_URL}/transaction/verify/{tx_ref}"
        print(f"🔍 Calling Chapa verify URL: {url}")
        resp = await client.get(
            url,
            headers={"Authorization": f"Bearer {secret_key}"},
        )
        print(f"🔍 Chapa verify response status: {resp.status_code}")
        print(f"🔍 Chapa verify response body: {resp.text[:500]}")
        if resp.status_code != 200:
            raise HTTPException(502, f"Chapa verify API error: {resp.text[:200]}")
        data = resp.json()
        chapa_status = data.get("data", {}).get("status", "")
        return {"status": chapa_status, "data": data}

@app.get("/api/v1/chapa/verify/{tx_ref}")
async def chapa_verify_path(tx_ref: str):
    """
    Path-based verification (used by frontend polling).
    """
    print(f"🔍 Received verify request for tx_ref: {tx_ref}")
    try:
        chapa = _get_chapa()
        secret_key = chapa.secret_key
        if not secret_key:
            print("❌ Chapa secret key not configured")
            raise HTTPException(500, "Chapa secret key not configured")

        url = f"{chapa.CHAPA_API_URL}/transaction/verify/{tx_ref}"
        print(f"🔍 Calling Chapa verify URL: {url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {secret_key}"},
            )
            print(f"🔍 Chapa verify response status: {resp.status_code}")
            print(f"🔍 Chapa verify response body: {resp.text[:500]}")

        if resp.status_code != 200:
            error_body = resp.text[:200]
            print(f"❌ Chapa verify error {resp.status_code}: {error_body}")
            return {"status": "failed", "error": f"Chapa API error: {error_body}"}

        data = resp.json()
        chapa_status = data.get("data", {}).get("status", "")
        print(f"🔍 Chapa status: {chapa_status}")

        if chapa_status == "success":
            result = await chapa.process_webhook({"tx_ref": tx_ref, "status": "success"})
            print(f"✅ Processed success for {tx_ref}")
            return {"status": "success", **result}
        elif chapa_status == "pending":
            return {"status": "pending", "tx_ref": tx_ref}
        else:
            return {"status": "failed", "tx_ref": tx_ref, "chapa_status": chapa_status}
    except Exception as e:
        print(f"❌ Exception in chapa_verify_path: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))

@app.get("/api/v1/chapa/verify")
async def chapa_verify_query(tx_ref: str = None):
    """
    Query‑based verification (used by Chapa redirect after payment).
    """
    if not tx_ref:
        # Return a simple HTML page explaining the error
        return HTMLResponse(content="""
        <html>
            <body style="font-family: sans-serif; text-align: center; padding: 2rem;">
                <h1>Payment verification in progress</h1>
                <p>Missing transaction reference. Please return to the app and wait for confirmation.</p>
                <p>If this persists, contact support.</p>
            </body>
        </html>
        """, status_code=400)

    # Forward to the path-based handler
    return await chapa_verify_path(tx_ref)
@app.post("/api/v1/chapa/webhook")
async def chapa_webhook(req: Request):
    chapa = _get_chapa()
    webhook_data = await chapa.verify_webhook(req)   # no second argument
    result = await chapa.process_webhook(webhook_data)
    return result

@app.get("/api/v1/user/{user_id}/balance")
async def get_balance(user_id: str):
    res = supabase.table("profiles").select("credit_balance").eq("id", user_id).execute()
    if not res.data:
        raise HTTPException(404, "User not found")
    balance = res.data[0].get("credit_balance", 0)
    return {"user_id": user_id, "credit_balance": balance}

@app.post("/api/v1/suno/callback")
async def suno_callback(request: Request):
    """
    Callback endpoint for Suno API.
    Logs the callback data and returns 200.
    """
    data = await request.json()
    print(f"🎵 Suno callback received: {data}")
    return {"status": "ok"}

@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run."""
    return {"status": "ok"}