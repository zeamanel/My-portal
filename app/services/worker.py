import os
import google.generativeai as genai
from supabase import create_client
from dotenv import load_dotenv

# Load your .env variables (Supabase URL, Keys, and Google API Key)
load_dotenv()

# Initialize Clients
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-3-flash')

def get_unindexed_images():
    """Finds images in storage that aren't in the context_images table yet."""
    # 1. Fetch all files from 'oda-brain/ET' bucket
    res = supabase.storage.from_('oda-brain').list('ET')
    storage_files = [f['name'] for f in res]
    
    # 2. Fetch all existing image_urls from context_images table
    db_res = supabase.table('context_images').select('image_url').execute()
    db_urls = [row['image_url'] for row in db_res.data]
    
    # 3. Identify missing files
    base_url = f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/oda-brain/ET/"
    unindexed = [f for f in storage_files if (base_url + f) not in db_urls]
    
    return unindexed

def run_librarian(filename):
    """Uses Gemini 3 Flash to describe the image and save to SQL."""
    full_url = f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/oda-brain/ET/{filename}"
    
    # Prompt for high-fidelity cultural analysis
    prompt = f"Analyze this image: {full_url}. Provide a high-fidelity 3-line description focusing on architectural textures, lighting, and specific Ethiopian cultural markers for an AI prompt optimizer."
    
    response = model.generate_content(prompt)
    description = response.text
    
    # Insert into context_images
    data = {
        "country_code": "ET",
        "folder": "auto-indexed",
        "image_url": full_url,
        "description": description
    }
    supabase.table('context_images').insert(data).execute()
    print(f"✅ Indexed: {filename}")
    return description

def run_creative_partner(description):
    """Generates a new 'Pinterest-style' template based on the new context."""
    prompt = f"Based on this image description: '{description}', create a viral AI image generation template. Output a JSON with: 'title', 'category', and 'hidden_enhancement' (a prompt string that adds artistic flair)."
    
    response = model.generate_content(prompt)
    # Note: In production, use response.text and a JSON parser
    print(f"💡 New Template Idea Generated: {response.text[:50]}...")

if __name__ == "__main__":
    print("🌙 OdaFlux Nightly Worker Started...")
    
    new_files = get_unindexed_images()
    if not new_files:
        print("☕ No new images to process. Everything is up to date.")
    else:
        for file in new_files:
            desc = run_librarian(file)
            run_creative_partner(desc)
            
    print("🏁 Worker Task Complete.")