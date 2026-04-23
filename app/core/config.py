import os
from dotenv import load_dotenv

# Load env vars from .env file
load_dotenv()

class Config:
    # Supabase (Frontend)
    SUPABASE_PROJECT_ID = os.getenv("VITE_SUPABASE_PROJECT_ID")
    SUPABASE_URL = os.getenv("VITE_SUPABASE_URL")
    SUPABASE_PUBLISHABLE_KEY = os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY")
    
    # Supabase (Backend - PRIVATE)
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
    SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "ai-generations")
    
    # Security
    JWT_SECRET = os.getenv("JWT_SECRET")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    SECRET_KEY = os.getenv("SECRET_KEY")
    
    # AI APIs
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    FAL_API_KEY = os.getenv("FAL_API_KEY")
    SUNO_API_KEY = os.getenv("SUNO_API_KEY")
    HIGGSFIELD_API_KEY = os.getenv("HIGGSFIELD_API_KEY")
    LAOZHANG_API_KEY = os.getenv("LAOZHANG_API_KEY")
    
    # App Settings
    DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "1K")
    DEFAULT_COST = int(os.getenv("DEFAULT_COST", 5))
    
    # Validation
    @classmethod
    def validate(cls):
        required = [
            'SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY',
            'OPENROUTER_API_KEY', 'GEMINI_API_KEY'
        ]
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing required env vars: {missing}")
