#!/usr/bin/env python3
"""
Test brochure generation with structured JSON using Gemini models.

Prerequisites:
    - Backend must support 'text_model' and 'image_model' in the request.
    - Gemini 3.1 Pro Preview and Gemini 3.1 Flash Image Preview must be available.
"""

import requests
import json

# Configuration
PORTAL_URL = "https://my-portal-760742977917.us-central1.run.app/api/brochure/generate"
API_KEY = "0112750803"  # Must match BROCHURE_API_KEY in Cloud Run

# Test payload with latest Gemini models
payload = {
    "design_style": "amharic-overlay",          # Must exist in design_templates
    "topic": "የአርቲፊሻል ኢንተሊጀንስ ጥቅሞች ለንግድ ድርጅቶች",
    "target_audience": "ትናንሽ ንግድ ባለቤቶች",
    "tone": "አሳማኝ – ጽሑፉን በአማርኛ ጻፍ",
    "text_model": "gemini-3.1-pro-preview",     # Latest Pro for better Amharic
    "include_image": True,
    "image_model": "gemini-3.1-flash-image-preview"  # Latest Flash image model
}

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json; charset=utf-8"
}

print(f"Sending request to {PORTAL_URL}")
print(f"Text model: {payload['text_model']}")
print(f"Image model: {payload['image_model']}")
print("Waiting for response (this may take 20-40 seconds)...")

try:
    response = requests.post(PORTAL_URL, headers=headers, json=payload, timeout=120)

    if response.status_code == 200:
        output_file = "structured_brochure.pdf"
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✅ PDF saved as {output_file}")
        pdf_url = response.headers.get("X-PDF-URL")
        if pdf_url:
            print(f"📄 Also available at: {pdf_url}")
    else:
        print(f"❌ Error {response.status_code}:")
        print(response.text[:1000])
        # If the backend returns a JSON error, try to parse it
        if response.headers.get("content-type") == "application/json":
            try:
                error_detail = response.json().get("detail")
                if error_detail:
                    print(f"Detail: {error_detail}")
            except:
                pass
except requests.exceptions.Timeout:
    print("❌ Request timed out (120 seconds). The generation may take longer.")
except Exception as e:
    print(f"❌ Unexpected error: {e}")