# Lulit AI Gateway

Universal AI generation gateway with economic checks and multi-provider routing.

## Features
- 🛡️ **Gatekeeper v2.0** - Enhanced generation pipeline
- 🌍 **Context Tune** - Regional context integration  
- 🤖 **Auto Model Selection** - Gemini 2.0 Flash powered
- 💰 **Economic System** - Credit-based with 70/30 creator splits
- 🔌 **Multi-provider** - OpenRouter, Fal.ai, Google Gemini

## Project Structure
\\\
Lulit/
├── app/
│   ├── gatekeeper/     # Economic and intent detection
│   ├── services/       # AI services and routing
│   ├── core/          # Configuration and prompts
│   └── main.py        # FastAPI server
├── sql/               # Database migrations
├── tests/             # Test files
└── verify_ai_models.py # Model verification
\\\

## Quick Start
1. Clone repository: \git clone <repo-url>\
2. Install dependencies: \pip install -r requirements.txt\
3. Configure \.env\ file (copy from \.env.example\)
4. Run server: \uvicorn app.main:app --reload --host 0.0.0.0 --port 8000\

## API Endpoints
- \POST /api/v2/generate\ - Enhanced generation with context
- \POST /api/v1/generate\ - Legacy gatekeeper
- \GET /api/v1/models/gated\ - Available models with validation
- \GET /api/v2/context/folders/{country}\ - Context folders

## Environment Variables
\\\
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
GEMINI_API_KEY=your-gemini-key
OPENROUTER_API_KEY=your-openrouter-key
FAL_API_KEY=your-fal-key
\\\

## Development
- FastAPI backend
- Supabase for database and storage
- Multi-AI provider integration
