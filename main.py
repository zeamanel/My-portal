import os
import re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from supabase import create_client, Client
from google import genai
from google.genai import types
import httpx
from dotenv import load_dotenv

load_dotenv()

# ==================== Supabase ====================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set. The app may not function correctly.")
    # Do NOT raise – let the app start anyway so we can debug.
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ==================== Gemini & OpenRouter ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ==================== FastAPI ====================
app = FastAPI(title="Odaflux Internal Portal", description="Admin & agent portal")

@app.get("/debug/env")
async def debug_env():
    import os
    return {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_SERVICE_ROLE_KEY": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "ALL_ENV_KEYS": list(os.environ.keys())
    }

templates = Jinja2Templates(directory="templates")

# ==================== Auth Dependency ====================
async def get_current_user(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        # If not an API call, maybe redirect to login page? We'll handle in HTML.
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user = supabase.auth.get_user(token)
        email = user.user.email
        res = supabase.table("authorized_users").select("id, role").eq("email", email).execute()
        if not res.data:
            raise HTTPException(status_code=403, detail="Access denied")
        return {"id": res.data[0]["id"], "email": email, "role": res.data[0]["role"]}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

async def admin_required(current_user=Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin rights required")
    return current_user

# ==================== Helper: Call LLM (Gemini or OpenRouter) ====================
async def call_llm(model_name: str, system_prompt: str, user_prompt: str, image_url: Optional[str] = None, image_base64: Optional[str] = None, temperature: float = 0.7) -> str:
    # Direct Gemini
    if model_name.startswith("gemini-"):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY missing")
        client = genai.Client(api_key=GEMINI_API_KEY)
        parts = [system_prompt + "\n\n" + user_prompt]
        if image_url:
            # download image
            async with httpx.AsyncClient() as http:
                resp = await http.get(image_url, timeout=30)
                resp.raise_for_status()
                img_bytes = resp.content
                mime = resp.headers.get("content-type", "image/jpeg")
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
        elif image_base64:
            # decode base64
            import base64
            img_data = base64.b64decode(image_base64)
            parts.append(types.Part.from_bytes(data=img_data, mime_type="image/jpeg"))
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=parts,
            config=types.GenerateContentConfig(temperature=temperature)
        )
        return response.text
    else:
        # OpenRouter
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY missing")
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
        if image_url:
            messages[0]["content"] = [
                {"type": "text", "text": system_prompt + "\n\n" + user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

# ==================== API Routes ====================
@app.get("/api/agents")
async def list_agents(user=Depends(get_current_user)):
    res = supabase.table("agent_prompts").select("*").execute()
    return {"agents": res.data}

@app.post("/api/run-agent")
async def run_agent(request: Request, user=Depends(get_current_user)):
    body = await request.json()
    image_url = body.get("image_url")
    agent_slug = body.get("agent_slug")
    human_context = body.get("human_context", "")
    if not image_url or not agent_slug:
        raise HTTPException(400, "Missing image_url or agent_slug")
    # fetch agent
    agent_res = supabase.table("agent_prompts").select("*").eq("slug", agent_slug).execute()
    if not agent_res.data:
        raise HTTPException(404, "Agent not found")
    agent = agent_res.data[0]
    model = agent["model_name"]
    system_prompt = agent["system_prompt"]
    user_prompt_template = agent.get("user_prompt_template", "")
    if user_prompt_template:
        user_prompt = user_prompt_template.replace("{human_context}", human_context)
    else:
        user_prompt = human_context
    try:
        analysis = await call_llm(model, system_prompt, user_prompt, image_url=image_url, temperature=agent.get("temperature", 0.7))
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(500, f"LLM error: {str(e)}")

@app.post("/api/save-annotation")
async def save_annotation(request: Request, user=Depends(get_current_user)):
    body = await request.json()
    image_url = body.get("image_url")
    human_context = body.get("human_context", "")
    ai_analysis = body.get("ai_analysis", "")
    agent_slug = body.get("agent_slug")
    if not image_url:
        raise HTTPException(400, "image_url required")
    # Insert into human_labels or user_annotations (choose table)
    # Using user_annotations (ensure table exists)
    data = {
        "image_url": image_url,
        "user_id": user["id"],
        "agent_slug": agent_slug,
        "human_context": human_context,
        "ai_analysis": ai_analysis,
        "created_at": "now()"
    }
    res = supabase.table("user_annotations").insert(data).execute()
    return {"status": "ok"}

@app.get("/api/image/next")
async def next_image(user=Depends(get_current_user)):
    # Get an image from oda-brain that hasn't been annotated yet by this user
    # Complex: we need to exclude already annotated images. Simplified: get a random unannotated.
    # We'll query user_annotations and exclude those image_urls.
    annotated = supabase.table("user_annotations").select("image_url").eq("user_id", user["id"]).execute()
    annotated_urls = [row["image_url"] for row in annotated.data] if annotated.data else []
    # We need a list of all image URLs in oda-brain. For simplicity, we can scan the bucket dynamically.
    # But that's heavy. Instead, assume we have a table `context_images` with all known image URLs.
    # Let's use context_images as source.
    query = supabase.table("context_images").select("image_url").limit(100)
    if annotated_urls:
        # Not in annotated_urls
        query = query.not_("image_url", "in", f"({','.join(annotated_urls)})")
    res = query.execute()
    if not res.data:
        return {"image": None}
    import random
    img = random.choice(res.data)
    return {"image": img}

# ==================== Template Factory Endpoints ====================
@app.post("/api/template/stage1")
async def stage1_generate_ideas(user=Depends(get_current_user)):
    system = "You are an expert product manager. Generate 5 fresh template ideas for an AI image/video generation platform, focusing on Ethiopian and African cultural themes. Return ONLY a valid JSON array with fields: title, description, media_type (Image/Video/Audio)."
    user_prompt = "Generate creative template ideas for the Ethiopian market."
    model = "gemini-3.1-flash-preview"
    try:
        ai_text = await call_llm(model, system, user_prompt, temperature=0.8)
        # extract JSON
        match = re.search(r'\[[\s\S]*\]', ai_text)
        if not match:
            raise ValueError("No JSON array found")
        ideas = eval(match.group())
        for idea in ideas:
            supabase.table("template_idea").insert({
                "stage": "raw",
                "title": idea.get("title"),
                "description": idea.get("description"),
                "prompt_template": idea.get("description"),
                "media_type": idea.get("media_type", "Image"),
                "raw_input_analysis": {"source": "ai_generation"}
            }).execute()
        return {"count": len(ideas)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/template/stage2")
async def stage2_convert_ideas(request: Request, user=Depends(get_current_user)):
    body = await request.json()
    idea_ids = body.get("idea_ids", [])
    if not idea_ids:
        raise HTTPException(400, "No idea IDs")
    import json as _json

    def extract_json_obj(text: str) -> dict:
        """Strip markdown fences then parse the first {...} block."""
        # Remove ```json ... ``` or ``` ... ``` fences
        text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '')
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            raise ValueError("No JSON object found in AI response")
        return _json.loads(match.group())

    res = supabase.table("template_idea").select("*").in_("id", idea_ids).execute()
    ideas = res.data
    created = 0
    errors = []
    for idea in ideas:
        try:
            # Pass 1: creative brief
            sys_prompt = "You are a senior creative director. Write a creative brief for this template idea."
            usr_prompt = f"Idea: {idea['title']}\n{idea['description']}\nMedia: {idea['media_type']}"
            brief = await call_llm("gemini-3.1-flash-preview", sys_prompt, usr_prompt, temperature=0.7)

            # Pass 2: structured template JSON
            sys2 = (
                "You are an AI prompt engineer. Return ONLY a valid JSON object (no markdown, no explanation) with these keys: "
                "title (string), description (string), prompt_template (string with {{placeholders}}), "
                "fields (array of objects with: field_name, field_label, field_type, placeholder, is_required)."
            )
            usr2 = f"Brief:\n{brief}\nMedia type: {idea['media_type']}"
            raw_json = await call_llm("gemini-3.1-flash-preview", sys2, usr2, temperature=0.5)

            enriched = extract_json_obj(raw_json)

            tpl_data = {
                "title": enriched.get("title") or idea["title"],
                "description": enriched.get("description") or idea.get("description", ""),
                "prompt_template": enriched.get("prompt_template") or idea.get("description", ""),
                "media_type": idea.get("media_type", "Image"),
                "is_published": False,
                "creator_id": "cf47617f-a205-46ab-9268-d8b816a54758",
                "base_price_tokens": 10
            }
            tpl_res = supabase.table("creator_templates").insert(tpl_data).select("id").execute()
            if tpl_res.data:
                template_id = tpl_res.data[0]["id"]
                for idx, field in enumerate(enriched.get("fields", [])):
                    supabase.table("template_fields").insert({
                        "template_id": template_id,
                        "field_name": field.get("field_name", f"field_{idx}"),
                        "field_label": field.get("field_label") or field.get("field_name", f"Field {idx}"),
                        "field_type": field.get("field_type", "text"),
                        "placeholder": field.get("placeholder", ""),
                        "is_required": bool(field.get("is_required", True)),
                        "display_order": idx
                    }).execute()
                supabase.table("template_idea").update({"stage": "processed"}).eq("id", idea["id"]).execute()
                created += 1
        except Exception as e:
            errors.append({"idea_id": idea.get("id"), "error": str(e)})

    return {"created": created, "errors": errors}

# ==================== Template CRUD & Test Endpoints ====================
@app.get("/api/templates")
async def list_templates(published: Optional[bool] = None, user=Depends(get_current_user)):
    query = supabase.table("creator_templates").select(
        "id, title, description, media_type, is_published, before_image_url, after_image_url, prompt_template, created_at"
    ).order("created_at", desc=True)
    if published is not None:
        query = query.eq("is_published", published)
    res = query.execute()
    return {"templates": res.data}

@app.get("/api/templates/{template_id}")
async def get_template(template_id: str, user=Depends(get_current_user)):
    tpl_res = supabase.table("creator_templates").select("*").eq("id", template_id).execute()
    if not tpl_res.data:
        raise HTTPException(404, "Template not found")
    fields_res = supabase.table("template_fields").select("*").eq("template_id", template_id).order("display_order").execute()
    return {"template": tpl_res.data[0], "fields": fields_res.data}

@app.patch("/api/templates/{template_id}")
async def update_template(template_id: str, request: Request, user=Depends(get_current_user)):
    data = await request.json()
    allowed = ["before_image_url", "after_image_url", "is_published", "title", "description"]
    update_data = {k: v for k, v in data.items() if k in allowed}
    if not update_data:
        raise HTTPException(400, "No valid fields to update")
    res = supabase.table("creator_templates").update(update_data).eq("id", template_id).execute()
    return {"template": res.data[0] if res.data else {}}

@app.post("/api/template/test")
async def test_template_generation(request: Request, user=Depends(get_current_user)):
    body = await request.json()
    template_id = body.get("template_id")
    field_values = body.get("field_values", {})
    model = body.get("model", "gemini-3.1-flash-preview")
    if not template_id:
        raise HTTPException(400, "template_id required")
    tpl_res = supabase.table("creator_templates").select("*").eq("id", template_id).execute()
    if not tpl_res.data:
        raise HTTPException(404, "Template not found")
    template = tpl_res.data[0]
    # Build final prompt by replacing {{placeholders}}
    prompt = template.get("prompt_template", "")
    for key, value in field_values.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", value)
    # Refine the prompt with LLM
    sys_prompt = "You are an expert AI image prompt engineer. Enhance this prompt for high-quality AI image generation. Return ONLY the enhanced prompt text, no labels or explanation."
    try:
        refined = await call_llm(model, sys_prompt, prompt, temperature=0.7)
    except Exception as e:
        refined = prompt  # fallback: return raw prompt
    return {"final_prompt": prompt, "refined_prompt": refined}

@app.get("/api/template/ideas")
async def list_ideas(stage: Optional[str] = None, user=Depends(get_current_user)):
    query = supabase.table("template_idea").select("*").order("created_at", desc=True)
    if stage:
        query = query.eq("stage", stage)
    res = query.execute()
    return {"ideas": res.data}

# ==================== Admin Endpoints ====================
@app.get("/api/admin/agents")
async def admin_list_agents(admin=Depends(admin_required)):
    res = supabase.table("agent_prompts").select("*").execute()
    return {"agents": res.data}

@app.post("/api/admin/agents")
async def admin_create_agent(request: Request, admin=Depends(admin_required)):
    data = await request.json()
    required = ["slug", "display_name", "system_prompt"]
    for f in required:
        if f not in data:
            raise HTTPException(400, f"Missing {f}")
    res = supabase.table("agent_prompts").insert(data).execute()
    return {"agent": res.data[0]}

@app.put("/api/admin/agents/{agent_id}")
async def admin_update_agent(agent_id: str, request: Request, admin=Depends(admin_required)):
    data = await request.json()
    res = supabase.table("agent_prompts").update(data).eq("id", agent_id).execute()
    return {"agent": res.data[0]}

@app.delete("/api/admin/agents/{agent_id}")
async def admin_delete_agent(agent_id: str, admin=Depends(admin_required)):
    supabase.table("agent_prompts").delete().eq("id", agent_id).execute()
    return {"status": "ok"}

@app.get("/api/admin/users")
async def admin_list_users(admin=Depends(admin_required)):
    res = supabase.table("authorized_users").select("*").execute()
    return {"users": res.data}

@app.post("/api/admin/users")
async def admin_create_user(request: Request, admin=Depends(admin_required)):
    data = await request.json()
    if "email" not in data:
        raise HTTPException(400, "Missing email")
    if "role" not in data:
        data["role"] = "worker"
    res = supabase.table("authorized_users").insert(data).execute()
    return {"user": res.data[0]}

@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: str, admin=Depends(admin_required)):
    supabase.table("authorized_users").delete().eq("id", user_id).execute()
    return {"status": "ok"}

# ==================== Page Routes ====================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/context-builder")
async def context_builder_page(request: Request):
    return templates.TemplateResponse("context_builder.html", {"request": request})

@app.get("/ocr")
async def ocr_page(request: Request):
    return templates.TemplateResponse("ocr.html", {"request": request})

@app.get("/ugc-analysis")
async def ugc_page(request: Request):
    return templates.TemplateResponse("ugc_analysis.html", {"request": request})

@app.get("/template-factory")
async def template_factory_page(request: Request):
    return templates.TemplateResponse("template_factory.html", {"request": request})

@app.get("/admin/agents")
async def admin_agents_page(request: Request):
    return templates.TemplateResponse("admin_agents.html", {"request": request})

@app.get("/admin/users")
async def admin_users_page(request: Request):
    return templates.TemplateResponse("admin_users.html", {"request": request})

@app.get("/gallery")
async def gallery_page(request: Request):
    return templates.TemplateResponse("gallery.html", {"request": request})

# ==================== static files (optional) ====================
from fastapi.staticfiles import StaticFiles
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== OCR / Context Images Endpoints ====================
import random
from fastapi import UploadFile, File, Form

@app.get("/api/ocr/images")
async def list_ocr_images(user=Depends(get_current_user)):
    """List all images from oda-brain bucket (EG, ET, KE, NG, VN) with processed status."""
    if not supabase:
        raise HTTPException(500, "Supabase not configured")
    # Get already processed image URLs from context_images
    processed_res = supabase.table("context_images").select("image_url").execute()
    processed_urls = {row["image_url"] for row in processed_res.data} if processed_res.data else set()
    
    bucket = supabase.storage.from_("oda-brain")
    all_images = []
    countries = ["EG", "ET", "KE", "NG", "VN"]
    
    for country in countries:
        try:
            # List all files recursively (simple approach – list top level and then subfolders)
            # We'll just list with prefix country and empty delimiter to get all files
            items = bucket.list(country, {"limit": 1000})
            # items are folders? Actually we need to recurse. Simpler: list with empty prefix and filter.
            # But for simplicity, we'll assume flat structure? No, your bucket has subfolders.
            # We'll implement a recursive helper.
            async def list_recursive(prefix):
                result = []
                files = bucket.list(prefix)
                for f in files:
                    if f["metadata"] is None:  # folder
                        result.extend(await list_recursive(f"{prefix}/{f['name']}"))
                    elif f["name"].lower().endswith(('.jpg','.jpeg','.png','.webp')):
                        public_url = bucket.get_public_url(f"{prefix}/{f['name']}")
                        # Extract folder: if prefix contains /, last part is folder; else default 'general'
                        parts = prefix.split('/')
                        folder = parts[-1] if len(parts) > 1 else 'general'
                        result.append({
                            "name": f["name"],
                            "url": public_url,
                            "country_code": country,
                            "folder": folder,
                            "processed": public_url in processed_urls
                        })
                return result
            all_images.extend(await list_recursive(country))
        except Exception as e:
            print(f"Error listing {country}: {e}")
    return {"images": all_images}

@app.post("/api/ocr/upload")
async def upload_image(
    user=Depends(get_current_user),
    file: UploadFile = File(...),
    country: str = Form(...),
    folder: str = Form(...),
    quality: float = Form(0.7),
    max_width: int = Form(1200)
):
    """Upload an image to oda-brain bucket (compressed) and return public URL."""
    if not supabase:
        raise HTTPException(500, "Supabase not configured")
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files allowed")
    # Read and compress image
    from PIL import Image
    import io
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    # Resize if needed
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    # Convert to RGB if needed (for JPEG)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    # Save compressed JPEG
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=int(quality * 100))
    output.seek(0)
    # Generate unique name
    import uuid
    ext = "jpg"
    filename = f"{uuid.uuid4()}.{ext}"
    path = f"{country}/{folder}/{filename}"
    # Upload
    bucket = supabase.storage.from_("oda-brain")
    bucket.upload(path, output.getvalue(), {"content-type": "image/jpeg"})
    public_url = bucket.get_public_url(path)
    return {"url": public_url, "path": path}

@app.post("/api/ocr/analyze")
async def analyze_images(request: Request, user=Depends(get_current_user)):
    """
    Analyze a list of image URLs with a chosen model and prompt.
    Body: { "image_urls": ["url1", ...], "model": "gemini-3.1-flash-preview", "prompt": "...", "save_to_db": true }
    """
    body = await request.json()
    image_urls = body.get("image_urls", [])
    model = body.get("model", "gemini-3.1-flash-preview")
    prompt = body.get("prompt", "Describe this image in detail for cultural context.")
    save_to_db = body.get("save_to_db", True)
    
    if not supabase:
        raise HTTPException(500, "Supabase not configured")
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Gemini API key missing")
    
    results = []
    for url in image_urls:
        # Find metadata for this URL (country, folder) – we need to fetch from the list endpoint or from URL path
        # For simplicity, we can extract from URL path: /storage/v1/object/public/oda-brain/ET/street-scenes/xxx.jpg
        import re
        match = re.search(r'/oda-brain/([^/]+)/([^/]+)/', url)
        country = match.group(1) if match else "unknown"
        folder = match.group(2) if match else "unknown"
        
        try:
            # Call Gemini vision
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=GEMINI_API_KEY)
            # Download image
            async with httpx.AsyncClient() as http:
                resp = await http.get(url, timeout=30)
                resp.raise_for_status()
                img_bytes = resp.content
                mime = resp.headers.get("content-type", "image/jpeg")
            parts = [types.Part.from_bytes(data=img_bytes, mime_type=mime)]
            response = await client.aio.models.generate_content(
                model=model,
                contents=parts + [prompt],
                config=types.GenerateContentConfig(temperature=0.4)
            )
            description = response.text.strip()
            # Generate embedding
            embed_response = await client.aio.models.embed_content(
                model="models/gemini-embedding-2",
                contents=description,
                config=types.EmbedContentConfig(output_dimensionality=3072)
            )
            embedding = embed_response.embeddings[0].values
            
            # Save to context_images if requested
            saved = False
            if save_to_db:
                # Check if exists
                existing = supabase.table("context_images").select("id").eq("image_url", url).execute()
                if not existing.data:
                    supabase.table("context_images").insert({
                        "image_url": url,
                        "country_code": country,
                        "folder": folder,
                        "description": description,
                        "embedding": embedding
                    }).execute()
                    saved = True
                else:
                    # Update
                    supabase.table("context_images").update({
                        "description": description,
                        "embedding": embedding
                    }).eq("image_url", url).execute()
                    saved = True
            results.append({
                "url": url,
                "description": description,
                "saved": saved
            })
        except Exception as e:
            results.append({
                "url": url,
                "error": str(e)
            })
    return {"results": results}

@app.post("/api/ocr/analyze-single")
async def analyze_single_image(request: Request, user=Depends(get_current_user)):
    """Analyze a single image with optional human annotation context."""
    body = await request.json()
    image_url = body.get("image_url")
    model = body.get("model", "gemini-3.1-flash-preview")
    base_prompt = body.get("prompt", "Describe this image in detail for cultural context.")
    human_context = body.get("human_context", "").strip()
    save_to_db = body.get("save_to_db", True)

    if not image_url:
        raise HTTPException(400, "image_url required")
    if not supabase or not GEMINI_API_KEY:
        raise HTTPException(500, "Missing configuration")

    # Country/folder: use explicit body values first, fall back to URL extraction
    url_match = re.search(r'/oda-brain/([^/]+)/([^/]+)/', image_url)
    country = body.get("country_code") or (url_match.group(1) if url_match else "unknown")
    folder  = body.get("folder")       or (url_match.group(2) if url_match else "general")

    # Build final prompt — append human annotation when provided
    if human_context:
        prompt = (
            f"{base_prompt}\n\n"
            f"The annotator has provided this additional context about the image:\n"
            f"\"{human_context}\"\n\n"
            f"Use this human knowledge to enrich your description."
        )
    else:
        prompt = base_prompt

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        async with httpx.AsyncClient() as http:
            resp = await http.get(image_url, timeout=30)
            resp.raise_for_status()
            img_bytes = resp.content
            mime = resp.headers.get("content-type", "image/jpeg")
        parts = [types.Part.from_bytes(data=img_bytes, mime_type=mime)]
        response = await client.aio.models.generate_content(
            model=model,
            contents=parts + [prompt],
            config=types.GenerateContentConfig(temperature=0.4)
        )
        description = response.text.strip()

        # Generate embedding
        embed_response = await client.aio.models.embed_content(
            model="models/gemini-embedding-2",
            contents=description,
            config=types.EmbedContentConfig(output_dimensionality=3072)
        )
        embedding = embed_response.embeddings[0].values

        saved = False
        if save_to_db:
            existing = supabase.table("context_images").select("id").eq("image_url", image_url).execute()
            if not existing.data:
                supabase.table("context_images").insert({
                    "image_url": image_url,
                    "country_code": country,
                    "folder": folder,
                    "description": description,
                    "embedding": embedding
                }).execute()
            else:
                supabase.table("context_images").update({
                    "description": description,
                    "embedding": embedding,
                    "country_code": country,
                    "folder": folder
                }).eq("image_url", image_url).execute()
            saved = True

        return {
            "description": description,
            "saved": saved,
            "country_code": country,
            "folder": folder,
            "image_url": image_url
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/annotation/recent")
async def recent_annotations(limit: int = 10, user=Depends(get_current_user)):
    """Return the most recently saved context_images rows."""
    res = supabase.table("context_images") \
        .select("id, image_url, country_code, folder, description, created_at") \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    return {"items": res.data}

# ==================== Gallery Endpoint ====================
@app.get("/api/gallery/images")
async def get_gallery_images(
    page: int = 1,
    per_page: int = 5,
    user=Depends(get_current_user)
):
    """
    Paginated list of images from context_images (or oda-brain if empty).
    Only returns image_url, country_code, folder, description.
    """
    if not supabase:
        raise HTTPException(500, "Supabase not configured")

    offset = (page - 1) * per_page
    # Fetch per_page+1 to detect whether a next page exists
    result_with_extra = supabase.table("context_images") \
        .select("image_url, country_code, folder, description") \
        .range(offset, offset + per_page) \
        .execute()

    items = result_with_extra.data
    has_next = len(items) > per_page
    images = items[:per_page]

    return {
        "images": images,
        "page": page,
        "has_next": has_next,
        "has_prev": page > 1
    }

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)