import uuid
import os
from datetime import datetime
from supabase import create_client

def upload_to_supabase_storage(supabase_client, user_id: str, image_bytes: bytes, 
                              model: str = "unknown", file_format: str = "png") -> str:
    """
    Upload image to Supabase Storage in user folder.
    Returns public URL of the uploaded image.
    """
    
    # Get bucket from environment or use default
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "ai-generations")
    
    # Create unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}.{file_format}"
    
    # Create user folder path: user_id/model/filename
    # Example: fc6f2b43-739c-4e8e-9e67-bd2389288922/flux.2-pro/20240101_120000_abc123.png
    storage_path = f"{user_id}/{model}/{filename}"
    
    try:
        # Upload to Supabase Storage
        supabase_client.storage.from_(bucket).upload(
            storage_path,
            image_bytes,
            file_options={"content-type": f"image/{file_format}"}
        )
        
        # Get public URL
        public_url = supabase_client.storage.from_(bucket).get_public_url(storage_path)
        
        print(f"   [FILE] Saved to: {storage_path}")
        print(f"   [LINK] Public URL: {public_url}")
        
        return public_url
        
    except Exception as e:
        print(f"   [WARN] Failed to upload to storage: {e}")
        # Fallback: return a placeholder or save locally
        return f"storage://{bucket}/{storage_path}"

def ensure_user_folder(supabase_client, user_id: str, model: str = ""):
    """
    Ensure user folder exists in storage (Supabase creates folders automatically on upload)
    """
    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "ai-generations")
    
    # Supabase doesn't require creating folders manually
    # They're created automatically when you upload to a path
    print(f"   📂 User folder: {user_id}/")
    if model:
        print(f"   🤖 Model subfolder: {model}/")
    
    return True
