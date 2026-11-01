import requests
import json

url = "https://my-portal-1045664496839.us-central1.run.app/api/brochure/generate"
headers = {
    "X-API-Key": "0112750803",
    "Content-Type": "application/json; charset=utf-8"
}
payload = {
    "design_style": "amharic-overlay",
    "topic": "የአርቲፊሻል ኢንተሊጀንስ ጥቅሞች ለንግድ ድርጅቶች",
    "target_audience": "ትናንሽ ንግድ ባለቤቶች",
    "tone": "አሳማኝ – ጽሑፉን በአማርኛ ጻፍ",
    "include_image": True,
    "image_model": "black-forest-labs/flux.2-pro"
}

response = requests.post(url, headers=headers, json=payload)
if response.status_code == 200:
    with open("amharic_brochure.pdf", "wb") as f:
        f.write(response.content)
    print("PDF saved successfully.")
else:
    print(f"Error {response.status_code}: {response.text}")