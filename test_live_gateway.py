"""Smoke-test the live gateway on Cloud Run.

Credentials come from the environment, never from this file:
  PORTAL_APP_KEY        an app's X-App-Key (e.g. Ethiogram's)
  PORTAL_ADMIN_EMAIL    admin login for the stats/apps checks
  PORTAL_ADMIN_PASSWORD
  PORTAL_BASE_URL       optional, defaults to the production URL
"""
import os
import sys
import httpx

BASE_URL = os.getenv("PORTAL_BASE_URL", "https://my-portal-1045664496839.us-central1.run.app")


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"Set {name} first.")
    return value


def test_live():
    print("Testing Live Central Chapa Gateway on Cloud Run...")

    # 1. Test payments page
    r = httpx.get(f"{BASE_URL}/payments", timeout=15)
    print("1. GET /payments status:", r.status_code)

    # 2. Client app payment initiation with API key
    payload = {
        "amount_etb": 200,
        "user_id": "ethio_user_77",
        "email": "customer@gmail.com"
    }
    headers = {"X-App-Key": _env("PORTAL_APP_KEY")}
    r = httpx.post(f"{BASE_URL}/api/v1/payments/initiate", json=payload, headers=headers, timeout=20)
    print("2. POST /api/v1/payments/initiate status:", r.status_code)
    data = r.json()
    print("   -> tx_ref:", data.get("tx_ref"))
    print("   -> checkout_url:", data.get("checkout_url"))
    tx_ref = data.get("tx_ref")

    # 3. Verify endpoint
    if tx_ref:
        r = httpx.get(f"{BASE_URL}/api/v1/payments/verify/{tx_ref}", timeout=20)
        print("3. GET /api/v1/payments/verify status:", r.status_code, r.json().get("status"))

    # 4. Admin stats via JWT
    login_resp = httpx.post(
        f"{BASE_URL}/api/auth/login",
        json={"email": _env("PORTAL_ADMIN_EMAIL"), "password": _env("PORTAL_ADMIN_PASSWORD")},
        timeout=15,
    )
    token = login_resp.json().get("access_token")
    r = httpx.get(f"{BASE_URL}/api/admin/payments/stats", headers={"Authorization": f"Bearer {token}"}, timeout=15)
    stats = r.json()
    print("4. GET /api/admin/payments/stats:", r.status_code)
    for row in stats.get("by_app", []):
        print(f"   * {row['name']}: {row['completed_etb']:,.2f} ETB collected, "
              f"{row['completed_count']}/{row['total_transactions']} completed")

    # 5. Apps list (keys are not printed)
    r = httpx.get(f"{BASE_URL}/api/admin/payments/apps", headers={"Authorization": f"Bearer {token}"}, timeout=15)
    apps = r.json().get("apps", [])
    print("5. Connected apps count:", len(apps))
    for a in apps:
        print(f"   * {a['name']} | ID: {a['app_id']}")


if __name__ == "__main__":
    test_live()
