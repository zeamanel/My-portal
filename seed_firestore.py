import csv
import os
import uuid
import datetime
from gcp_client import get_firestore_client, hash_password

def seed_database():
    db = get_firestore_client()
    print("Seeding Firestore database for betcare...")

    # 1. Seed Authorized Users
    users_ref = db.collection("authorized_users")
    default_users = [
        {"email": "betcareplc@gmail.com", "role": "admin", "password": "Password123!"},
        {"email": "xeamanel@gmail.com", "role": "admin", "password": "Password123!"},
    ]
    for u in default_users:
        doc_id = u["email"].replace("@", "_at_").replace(".", "_dot_")
        existing = users_ref.document(doc_id).get()
        if not existing.exists:
            users_ref.document(doc_id).set({
                "id": str(uuid.uuid4()),
                "email": u["email"],
                "role": u["role"],
                "password_hash": hash_password(u["password"]),
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })
            print(f"  + Added user: {u['email']} (role: {u['role']})")
        else:
            print(f"  * User exists: {u['email']}")

    # 2. Seed Design Templates
    dt_ref = db.collection("design_templates")
    default_templates = [
        {
            "id": "amharic-overlay",
            "style_key": "amharic-overlay",
            "layout_hint": "Modern centered title with elegant typography, Amharic font support, and striking hero image background",
            "html_template": """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Ethiopic:wght@400;700&display=swap');
body {
    font-family: 'Noto Sans Ethiopic', 'Segoe UI', Tahoma, sans-serif;
    margin: 0;
    padding: 30px;
    background: #f8fafc;
    color: #1e293b;
}
.header {
    text-align: center;
    padding: 20px 0;
    border-bottom: 2px solid #0284c7;
    margin-bottom: 25px;
}
h1 { font-size: 26px; color: #0f172a; margin-bottom: 8px; }
.tagline { font-size: 16px; color: #0284c7; font-weight: bold; }
.hero-image {
    width: 100%;
    max-height: 280px;
    object-fit: cover;
    border-radius: 8px;
    margin-bottom: 20px;
}
.intro { font-size: 14px; line-height: 1.6; margin-bottom: 20px; }
.features {
    background: #ffffff;
    padding: 15px 20px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    margin-bottom: 25px;
}
.features ul { margin: 0; padding-left: 20px; }
.features li { font-size: 13px; line-height: 1.6; margin-bottom: 8px; }
.cta {
    background: #0284c7;
    color: #ffffff;
    text-align: center;
    padding: 15px;
    border-radius: 8px;
    font-size: 15px;
    font-weight: bold;
}
{{ css }}
</style>
</head>
<body>
<div class="header">
    <h1>{{ headline or title }}</h1>
    <div class="tagline">{{ tagline }}</div>
</div>
{% if image_url %}
<img class="hero-image" src="{{ image_url }}" alt="Brochure visual">
{% endif %}
<div class="intro">
    <p>{{ intro }}</p>
</div>
{% if features %}
<div class="features">
    <ul>
    {% for f in features %}
        <li>{{ f }}</li>
    {% endfor %}
    </ul>
</div>
{% endif %}
{% if cta %}
<div class="cta">{{ cta }}</div>
{% endif %}
</body>
</html>""",
            "css": "",
            "schema": [
                {"name": "headline", "type": "str", "required": True},
                {"name": "tagline", "type": "str", "required": False},
                {"name": "intro_paragraph", "type": "str", "required": True},
                {"name": "features", "type": "List[str]", "required": True},
                {"name": "cta_text", "type": "str", "required": True}
            ]
        }
    ]
    for dt in default_templates:
        existing = dt_ref.document(dt["id"]).get()
        if not existing.exists:
            dt_ref.document(dt["id"]).set(dt)
            print(f"  + Added design template: {dt['style_key']}")
        else:
            print(f"  * Design template exists: {dt['style_key']}")

    # 3. Seed AI Models
    models_ref = db.collection("ai_models")
    default_models = [
        {"model_id": "gemini-2.5-pro", "display_name": "Gemini 2.5 Pro", "media_type": "Image", "provider": "Google", "service_type": "genai", "is_active": True, "show_in_creator": True},
        {"model_id": "gemini-3.1-pro-preview", "display_name": "Gemini 3.1 Pro Preview", "media_type": "Image", "provider": "Google", "service_type": "genai", "is_active": True, "show_in_creator": True},
        {"model_id": "gemini-3.1-flash-image-preview", "display_name": "Gemini 3.1 Flash Image", "media_type": "Image", "provider": "Google", "service_type": "genai", "is_active": True, "show_in_creator": True},
        {"model_id": "black-forest-labs/flux.2-pro", "display_name": "Flux 2 Pro", "media_type": "Image", "provider": "Black Forest Labs", "service_type": "fal", "is_active": True, "show_in_creator": True},
    ]
    for m in default_models:
        existing = models_ref.document(m["model_id"].replace("/", "_")).get()
        if not existing.exists:
            models_ref.document(m["model_id"].replace("/", "_")).set(m)
            print(f"  + Added AI model: {m['display_name']}")
        else:
            print(f"  * AI model exists: {m['display_name']}")

    # 4. Seed Creator Templates from CSV if available
    csv_path = r"C:\Users\User\Desktop\creator_templates_rows.csv"
    if os.path.exists(csv_path):
        print(f"Loading creator templates from {csv_path}...")
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                t_id = row.get("id")
                if not t_id:
                    continue
                db.collection("creator_templates").document(t_id).set(dict(row))
                count += 1
            print(f"  + Imported {count} creator templates from CSV!")

    # 5. Seed Template Fields from CSV if available
    fields_csv_path = r"C:\Users\User\Desktop\template_fields_rows.csv"
    if os.path.exists(fields_csv_path):
        print(f"Loading template fields from {fields_csv_path}...")
        with open(fields_csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                f_id = row.get("id") or str(uuid.uuid4())
                db.collection("template_fields").document(f_id).set(dict(row))
                count += 1
            print(f"  + Imported {count} template fields from CSV!")

    print("Firestore seeding completed successfully!")

if __name__ == "__main__":
    seed_database()
