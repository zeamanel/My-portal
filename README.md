# Internal Portal – My‑Portal

A secure internal web portal for image annotation, OCR, content analysis, template building, and admin management.  
Uses Supabase for data and authentication, Gemini/OpenRouter for AI.

**Deployed as a separate Cloud Run service** – independent from any main backend.

---

## ✨ Features

- Role‑based access (admin / worker) – only authorised emails can log in.
- **Agent library** – store reusable AI prompts (system prompt, model, temperature) in a database table.
- **Context Builder** – annotate images with human knowledge + AI analysis.
- **OCR page** – extract text from images.
- **UGC Analysis** – analyse user‑generated content.
- **Template Factory** – two‑stage pipeline: generate raw ideas → convert selected into templates.
- **Admin panels** – manage agents and authorised users.
- All AI calls are **server‑side** – API keys never exposed to the browser.

---

## 🗄️ Database Setup (Supabase)

Run these SQL commands in your Supabase SQL Editor **in order**.

### 1. Authorised users table

```sql
CREATE TABLE authorized_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL DEFAULT 'worker',   -- 'admin' or 'worker'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert yourself as admin
INSERT INTO authorized_users (email, role) VALUES ('your-email@example.com', 'admin');