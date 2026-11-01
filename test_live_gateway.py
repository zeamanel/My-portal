import httpx

BASE_URL = "https://my-portal-1045664496839.us-central1.run.app"

def test_live():
    print("Testing Live Central Chapa Gateway on Cloud Run...")
    
    # 1. Test payments page
    r = httpx.get(f"{BASE_URL}/payments", timeout=15)
    print("1. GET /payments status:", r.status_code)
    
    # 2. Test Client App (Ethiogram) Payment Initiation with API key
    payload = {
        "amount_etb": 200,
        "user_id": "ethio_user_77",
        "email": "customer@gmail.com"
    }
    headers = {"X-App-Key": "chp_live_ethiogram_4e9b210c8a7f"}
    r = httpx.post(f"{BASE_URL}/api/v1/payments/initiate", json=payload, headers=headers, timeout=20)
    print("2. POST /api/v1/payments/initiate status:", r.status_code)
    data = r.json()
    print("   -> tx_ref:", data.get("tx_ref"))
    print("   -> checkout_url:", data.get("checkout_url"))
    tx_ref = data.get("tx_ref")
    
    # 3. Test Verify endpoint
    if tx_ref:
        r = httpx.get(f"{BASE_URL}/api/v1/payments/verify/{tx_ref}", timeout=20)
        print("3. GET /api/v1/payments/verify status:", r.status_code, r.json().get("status"))

    # 4. Test Admin stats via JWT
    login_resp = httpx.post(f"{BASE_URL}/api/auth/login", json={"email": "betcareplc@gmail.com", "password": "Password123!"}, timeout=15)
    token = login_resp.json().get("access_token")
    r = httpx.get(f"{BASE_URL}/api/admin/payments/stats", headers={"Authorization": f"Bearer {token}"}, timeout=15)
    print("4. GET /api/admin/payments/stats:", r.status_code, r.json())

    # 5. Test Apps list
    r = httpx.get(f"{BASE_URL}/api/admin/payments/apps", headers={"Authorization": f"Bearer {token}"}, timeout=15)
    apps = r.json().get("apps", [])
    print("5. Connected apps count:", len(apps))
    for a in apps:
        print(f"   * {a['name']} | ID: {a['app_id']} | Key: {a['api_key']}")

if __name__ == "__main__":
    test_live()
