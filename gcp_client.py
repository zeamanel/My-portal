import os
import hashlib
import hmac
import secrets
import datetime
import subprocess
import uuid
from typing import Optional, List, Dict, Any

import jwt
from google.cloud import firestore, storage
from google.oauth2.credentials import Credentials
import google.auth

GCP_PROJECT = os.getenv("GCP_PROJECT", "betcare")
ODA_BRAIN_BUCKET = os.getenv("ODA_BRAIN_BUCKET", "betcare-oda-brain")
BROCHURES_BUCKET = os.getenv("BROCHURES_BUCKET", "betcare-brochures")
JWT_SECRET = os.getenv("JWT_SECRET", "odaflux-portal-jwt-secret-2026-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7 days

_firestore_client = None
_storage_client = None

def _get_credentials():
    """Get GCP credentials.
    On Cloud Run (where K_SERVICE is set), use Google Auth Default.
    Locally, use gcloud CLI token so it always uses the active gcloud account.
    """
    if os.getenv("K_SERVICE") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            creds, _ = google.auth.default()
            return creds
        except Exception as e:
            print(f"Warning: google.auth.default() failed: {e}")

    # Local development fallback: get token directly from active gcloud CLI
    try:
        token = subprocess.check_output("gcloud auth print-access-token", shell=True).decode().strip()
        if token:
            return Credentials(token)
    except Exception as e:
        print(f"Warning: Failed to obtain credentials via gcloud CLI: {e}")

    try:
        creds, _ = google.auth.default()
        return creds
    except Exception:
        return None

def get_firestore_client() -> firestore.Client:
    global _firestore_client
    if _firestore_client is None:
        creds = _get_credentials()
        if creds:
            _firestore_client = firestore.Client(project=GCP_PROJECT, credentials=creds)
        else:
            _firestore_client = firestore.Client(project=GCP_PROJECT)
    return _firestore_client

def get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        creds = _get_credentials()
        if creds:
            _storage_client = storage.Client(project=GCP_PROJECT, credentials=creds)
        else:
            _storage_client = storage.Client(project=GCP_PROJECT)
    return _storage_client

# ==================== Auth Helpers ====================
def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hash password using PBKDF2-HMAC-SHA256."""
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return f"{salt}:{key.hex()}"

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hashed string."""
    try:
        if not hashed or ":" not in hashed:
            return False
        salt, expected_hex = hashed.split(":", 1)
        actual_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return hmac.compare_digest(actual_key.hex(), expected_hex)
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.now(datetime.timezone.utc) + expires_delta
    else:
        expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate JWT access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except Exception:
        return None

# ==================== Supabase Compatibility Query & Storage Adapter ====================
class QueryResult:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

class StorageBucketAdapter:
    def __init__(self, bucket_name: str):
        # Map logical names to real GCS buckets
        if bucket_name == "oda-brain":
            self.bucket_name = ODA_BRAIN_BUCKET
        elif bucket_name == "brochures":
            self.bucket_name = BROCHURES_BUCKET
        else:
            self.bucket_name = bucket_name

    def upload(self, destination_blob: str, data: Any, file_options: Optional[dict] = None) -> dict:
        client = get_storage_client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob)
        content_type = (file_options or {}).get("content-type", "application/octet-stream")
        if isinstance(data, (bytes, bytearray)):
            blob.upload_from_string(data, content_type=content_type)
        elif hasattr(data, "read"):
            blob.upload_from_file(data, content_type=content_type)
        else:
            blob.upload_from_string(str(data), content_type=content_type)
        return {"Key": destination_blob}

    def get_public_url(self, filename: str) -> str:
        return f"https://storage.googleapis.com/{self.bucket_name}/{filename}"

class StorageAdapter:
    def from_(self, bucket_name: str) -> StorageBucketAdapter:
        return StorageBucketAdapter(bucket_name)

class FirestoreTableQuery:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.db = get_firestore_client()
        self.coll = self.db.collection(collection_name)
        self._filters = []
        self._orders = []
        self._limit_val = None
        self._selected_fields = None
        self._operation = "select"
        self._write_data = None
        self._not_in_filter = None

    def select(self, fields: str = "*"):
        if fields and fields != "*":
            self._selected_fields = [f.strip() for f in fields.split(",")]
        return self

    def eq(self, field: str, value: Any):
        self._filters.append((field, "==", value))
        return self

    def neq(self, field: str, value: Any):
        self._filters.append((field, "!=", value))
        return self

    def in_(self, field: str, values: list):
        if values:
            # Firestore supports 'in' up to 30 items
            self._filters.append((field, "in", values[:30]))
        else:
            # Empty list in filter matches nothing
            self._filters.append((field, "==", "__NO_MATCH__"))
        return self

    def order(self, field: str, desc: bool = False):
        direction = firestore.Query.DESCENDING if desc else firestore.Query.ASCENDING
        self._orders.append((field, direction))
        return self

    def limit(self, count: int):
        self._limit_val = count
        return self

    def not_(self, field: str, op: str, value: Any):
        # Handle exclusion (e.g. not in)
        if op.lower() == "in":
            # parse csv value or list
            if isinstance(value, str):
                vals = [v.strip(" ()'\"") for v in value.split(",")]
            else:
                vals = list(value)
            self._not_in_filter = (field, set(vals))
        return self

    def insert(self, data: Any):
        self._operation = "insert"
        self._write_data = data
        return self

    def update(self, data: dict):
        self._operation = "update"
        self._write_data = data
        return self

    def delete(self):
        self._operation = "delete"
        return self

    def execute(self) -> QueryResult:
        if self._operation == "insert":
            inserted = []
            items = self._write_data if isinstance(self._write_data, list) else [self._write_data]
            for item in items:
                doc_data = dict(item)
                doc_id = doc_data.get("id") or str(uuid.uuid4())
                doc_data["id"] = doc_id
                if "created_at" not in doc_data or doc_data["created_at"] == "now()":
                    doc_data["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self.coll.document(doc_id).set(doc_data)
                inserted.append(doc_data)
            return QueryResult(inserted)

        elif self._operation == "update":
            # Fetch matching documents to update
            docs = self._fetch_docs()
            for doc in docs:
                update_dict = dict(self._write_data)
                update_dict["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                doc.reference.update(update_dict)
            return QueryResult([{"id": d.id, **d.to_dict()} for d in docs])

        elif self._operation == "delete":
            docs = self._fetch_docs()
            for doc in docs:
                doc.reference.delete()
            return QueryResult([])

        else: # select
            docs = self._fetch_docs()
            results = []
            for d in docs:
                data = d.to_dict() or {}
                if "id" not in data:
                    data["id"] = d.id
                
                # Check not_in filter in memory if specified
                if self._not_in_filter:
                    field, excluded_set = self._not_in_filter
                    if data.get(field) in excluded_set:
                        continue

                # Filter fields if specific fields were requested
                if self._selected_fields:
                    filtered = {k: data.get(k) for k in self._selected_fields}
                    if "id" in data and "id" not in filtered:
                        filtered["id"] = data["id"]
                    results.append(filtered)
                else:
                    results.append(data)

            # In-memory ordering fallback for multiple orders
            for field, direction in reversed(self._orders):
                results.sort(
                    key=lambda x: (x.get(field) is None, x.get(field)),
                    reverse=(direction == firestore.Query.DESCENDING)
                )

            if self._limit_val:
                results = results[:self._limit_val]

            return QueryResult(results)

    def _fetch_docs(self):
        query = self.coll
        for field, op, val in self._filters:
            query = query.where(filter=firestore.FieldFilter(field, op, val))
        if self._limit_val and not self._not_in_filter:
            query = query.limit(self._limit_val)
        return list(query.stream())

class SupabaseGCPAdapter:
    def __init__(self):
        self.storage = StorageAdapter()

    def table(self, table_name: str) -> FirestoreTableQuery:
        return FirestoreTableQuery(table_name)
