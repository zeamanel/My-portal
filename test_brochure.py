#!/usr/bin/env python3
"""
Test brochure generation with Amharic text using Gemini model.

Usage:
    python test_brochure.py

Requirements:
    requests library (pip install requests)
"""

import requests
import sys

# ==================== CONFIGURATION ====================
PORTAL_URL = "https://my-portal-760742977917.us-central1.run.app/api/brochure/generate"
API_KEY = "0112750803"  # Must match BROCHURE_API_KEY in Cloud Run

# Test payload – Amharic topic, target audience, and tone
payload = {
    "design_style": "amharic-overlay",          # Must exist in design_templates
    "topic": "የአርቲፊሻል ኢንተሊጀንስ ጥቅሞች ለንግድ ድርጅቶች",
    "target_audience": "ትናንሽ ንግድ ባለቤቶች",
    "tone": "አሳማኝ – ጽሑፉን በአማርኛ ጻፍ",
    "text_model": "gemini-3.1-pro-preview",   # Or "gemini-2.0-flash"
    "include_image": True,
    "image_model": "gemini-3.1-flash-image-preview"
}

# Optional: test with English and a different model
# payload = {
#     "design_style": "modern",
#     "topic": "Benefits of AI for Small Business",
#     "target_audience": "small business owners",
#     "tone": "persuasive",
#     "text_model": "gemini-2.5-pro",
#     "include_image": False
# }

# ==================== REQUEST ====================
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json; charset=utf-8"
}

print(f"Sending request to {PORTAL_URL}")
print(f"Using text model: {payload['text_model']}")
print("Waiting for response (this may take 15-30 seconds)...")

try:
    response = requests.post(PORTAL_URL, headers=headers, json=payload, timeout=90)

    if response.status_code == 200:
        output_file = "amharic_gemini_brochure.pdf"
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✅ Success! PDF saved as {output_file}")
        # Optionally print the PDF URL if returned in headers
        pdf_url = response.headers.get("X-PDF-URL")
        if pdf_url:
            print(f"📄 PDF also available at: {pdf_url}")
    else:
        print(f"❌ Error {response.status_code}:")
        print(response.text[:500])  # show first 500 chars of error
        sys.exit(1)

except requests.exceptions.Timeout:
    print("❌ Request timed out (90 seconds). The generation may take longer.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)