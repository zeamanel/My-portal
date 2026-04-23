import os
import httpx
import json
import base64
from typing import Optional, Dict, Any, Union

from app.core.config import Config

class OpenRouterService:
    def __init__(self):
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            print("WARNING: OPENROUTER_API_KEY not found. OpenRouter calls will fail.")

    async def generate_via_openrouter(self, prompt: str, model_id: str = "black-forest-labs/flux.2-pro") -> bytes:
        """
        Generates an image via OpenRouter using chat completions endpoint.
        OpenRouter doesn't support /images/generations - use chat completions with images.
        """
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://musa.ai",
            "X-Title": "Musa AI",
            "Content-Type": "application/json"
        }

        # Use chat completions endpoint for image generation
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model_id,
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
            "max_tokens": 100,  # Some models need this
            "modalities": ["image", "text"]  # Explicitly request image output
        }

        async with httpx.AsyncClient() as client:
            try:
                print(f"🔄 Sending request to OpenRouter with model: {model_id}")
                print(f"📝 Prompt: {prompt[:100]}...")
                
                response = await client.post(url, headers=headers, json=payload, timeout=120.0)
                
                # Check for standard errors
                if response.status_code != 200:
                    error_text = response.text[:500]
                    print(f"❌ OpenRouter API Error ({response.status_code}): {error_text}")
                    
                    # Try to parse error
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('error', {}).get('message', error_text)
                        raise Exception(f"OpenRouter API Error ({response.status_code}): {error_detail}")
                    except:
                        raise Exception(f"OpenRouter API Error ({response.status_code}): {error_text}")

                data = response.json()
                print(f"✅ OpenRouter response received")
                
                # Debug: print response structure
                print(f"📊 Response keys: {list(data.keys())}")
                
                # OpenRouter returns images in the message content
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    print(f"📊 Choice keys: {list(choice.keys())}")
                    
                    if "message" in choice:
                        message = choice["message"]
                        print(f"📊 Message keys: {list(message.keys())}")
                        
                        # Check for images in different possible locations
                        if "images" in message and len(message["images"]) > 0:
                            images = message["images"]
                            print(f"🎨 Found {len(images)} image(s) in 'images' field")
                            
                            # Get the first image
                            image_data = images[0]
                            print(f"📊 Image data keys: {list(image_data.keys())}")
                            
                            if "image_url" in image_data:
                                image_url = image_data["image_url"]["url"]
                                print(f"🌐 Image URL format: {image_url[:50]}...")
                                
                                # Handle base64 or direct URL
                                if image_url.startswith("data:image/"):
                                    # Extract base64 data
                                    header, base64_data = image_url.split(",", 1)
                                    image_format = header.split("/")[1].split(";")[0]
                                    print(f"📸 Image format: {image_format}")
                                    
                                    image_bytes = base64.b64decode(base64_data)
                                    print(f"✅ Decoded base64 image: {len(image_bytes)} bytes")
                                    return image_bytes
                                else:
                                    # Download from URL
                                    print(f"⬇️ Downloading image from URL: {image_url}")
                                    image_response = await client.get(image_url)
                                    if image_response.status_code != 200:
                                        raise Exception(f"Failed to download image from URL: {image_url}")
                                        
                                    print(f"✅ Downloaded image: {len(image_response.content)} bytes")
                                    return image_response.content
                            else:
                                print(f"⚠️ No 'image_url' in image data")
                        else:
                            print(f"⚠️ No 'images' field in message, checking alternative locations...")
                            
                            # Some models might return image differently
                            if "content" in message:
                                content = message["content"]
                                print(f"📝 Message content type: {type(content)}")
                                
                                # If content is a list
                                if isinstance(content, list):
                                    for item in content:
                                        print(f"📊 Content item type: {item.get('type')}")
                                        if item.get('type') == 'image_url' and 'image_url' in item:
                                            image_url = item['image_url']['url']
                                            print(f"🌐 Found image in content list")
                                            
                                            if image_url.startswith("data:image/"):
                                                header, base64_data = image_url.split(",", 1)
                                                image_bytes = base64.b64decode(base64_data)
                                                return image_bytes
                                            else:
                                                image_response = await client.get(image_url)
                                                return image_response.content
                else:
                    print(f"⚠️ No choices in response, full response: {json.dumps(data, indent=2)[:500]}...")
                    
                # If we get here, we didn't find an image
                raise Exception(f"Unexpected response format from OpenRouter. No image found. Response: {json.dumps(data, indent=2)[:500]}...")

            except httpx.RequestError as e:
                print(f"❌ Network error calling OpenRouter: {e}")
                raise Exception(f"Network error calling OpenRouter: {e}")
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                raise

    async def generate_image_to_image(self, prompt: str, reference_image_url: str, model_id: str = "black-forest-labs/flux.2-pro") -> bytes:
        """
        Image-to-image generation with a reference image.
        """
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://musa.ai",
            "X-Title": "Musa AI",
            "Content-Type": "application/json"
        }

        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": reference_image_url
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100,
            "modalities": ["image", "text"]
        }

        async with httpx.AsyncClient() as client:
            try:
                print(f"🎨 Sending image-to-image request with reference: {reference_image_url}")
                response = await client.post(url, headers=headers, json=payload, timeout=120.0)
                
                if response.status_code != 200:
                    error_text = response.text[:500]
                    raise Exception(f"OpenRouter API Error ({response.status_code}): {error_text}")

                data = response.json()
                
                # Extract image from response (same as above)
                if "choices" in data and len(data["choices"]) > 0:
                    if "message" in data["choices"][0]:
                        message = data["choices"][0]["message"]
                        if "images" in message and len(message["images"]) > 0:
                            image_data = message["images"][0]
                            image_url = image_data["image_url"]["url"]
                            
                            if image_url.startswith("data:image/"):
                                header, base64_data = image_url.split(",", 1)
                                return base64.b64decode(base64_data)
                            else:
                                image_response = await client.get(image_url)
                                return image_response.content
                                
                raise Exception("No image found in response")
                
            except httpx.RequestError as e:
                raise Exception(f"Network error: {e}")