# app/routers/payments_dashboard.py
"""Admin-facing payment collection dashboard API.

Mounted by main.py with `dependencies=[Depends(admin_required)]`, so the admin
check lives with the rest of the app's auth rather than being duplicated here —
that also avoids a circular import back into main.

Aggregation happens in Postgres (see migrations/0002) rather than in Python:
PostgREST caps a plain select at 1000 rows, so summing client-side would
silently under-report once the ledger outgrows that.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.deps import APP_IDS, LEDGER_TIMEZONE, require_chapa, require_supabase
from app.services.ledger import TABLE, refund_transaction, reverify

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/payments", tags=["payments-dashboard"])

TZ = ZoneInfo(LEDGER_TIMEZONE)
STATUSES = ("pending", "success", "failed", "cancelled", "refunded")

LIST_COLUMNS = (
    "tx_ref, app_id, external_user_id, amount_etb, currency, status, "
    "chapa_ref, created_at, updated_at, succeeded_at, failed_at, refunded_at, "
    "failure_reason"
)


class RefundBody(BaseModel):
    amount_etb: Optional[float] = Field(default=None, gt=0)


def _window(days: int) -> Tuple[datetime, datetime]:
    """[start, end) covering the last `days` local days, ending tomorrow 00:00.

    Anchored to Ethiopian local midnight so "last 7 days" means seven business
    days as the owner experiences them, not seven UTC days.
    """
    now_local = datetime.now(TZ)
    end = (now_local + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return end - timedelta(days=days), end


def _rpc(sb, name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        return sb.rpc(name, params).execute().data or []
    except Exception as e:
        message = str(e)
        if "PGRST202" in message or "does not exist" in message or "404" in message:
            raise HTTPException(
                503,
                f"Aggregate function {name}() is missing. Run "
                f"app/migrations/0002_chapa_dashboard.sql in the my-portal "
                f"Supabase SQL editor, then: notify pgrst, 'reload schema';",
            )
        logger.exception("rpc %s failed", name)
        raise HTTPException(500, f"Aggregate query failed: {message}")


def _blank() -> Dict[str, Any]:
    return {
        "counts": {s: 0 for s in STATUSES},
        "amounts": {s: 0.0 for s in STATUSES},
    }


def _derive(app_id: str, bucket: Dict[str, Any]) -> Dict[str, Any]:
    counts, amounts = bucket["counts"], bucket["amounts"]
    # A refunded payment succeeded first, so it counts as settled, but its money
    # is no longer held — hence collected_etb excludes it.
    settled = counts["success"] + counts["failed"] + counts["cancelled"] + counts["refunded"]
    won = counts["success"] + counts["refunded"]
    return {
        "app_id": app_id,
        "collected_etb": round(amounts["success"], 2),
        "pending_etb": round(amounts["pending"], 2),
        "refunded_etb": round(amounts["refunded"], 2),
        "failed_etb": round(amounts["failed"], 2),
        "attempts": sum(counts.values()),
        "success_count": counts["success"],
        "pending_count": counts["pending"],
        "failed_count": counts["failed"],
        "cancelled_count": counts["cancelled"],
        "refunded_count": counts["refunded"],
        # Share of *settled* attempts that were paid. Pending is excluded, so an
        # idle day does not drag the rate down.
        "success_rate": round(won / settled, 4) if settled else None,
    }


@router.get("/summary")
async def summary(days: int = Query(30, ge=1, le=365)):
    sb = require_supabase()
    start, end = _window(days)
    params = {"p_from": start.isoformat(), "p_to": end.isoformat()}

    buckets: Dict[str, Dict[str, Any]] = {a: _blank() for a in APP_IDS}
    for row in _rpc(sb, "chapa_app_summary", params):
        bucket = buckets.setdefault(row["app_id"], _blank())
        status = row["status"]
        if status not in bucket["counts"]:
            bucket["counts"][status] = 0
            bucket["amounts"][status] = 0.0
        bucket["counts"][status] += int(row["txn_count"])
        bucket["amounts"][status] += float(row["total_etb"])

    by_app = [_derive(a, buckets[a]) for a in buckets]
    by_app.sort(key=lambda r: (-r["collected_etb"], r["app_id"]))

    totals = {
        "collected_etb": round(sum(r["collected_etb"] for r in by_app), 2),
        "pending_etb": round(sum(r["pending_etb"] for r in by_app), 2),
        "refunded_etb": round(sum(r["refunded_etb"] for r in by_app), 2),
        "attempts": sum(r["attempts"] for r in by_app),
        "success_count": sum(r["success_count"] for r in by_app),
        "pending_count": sum(r["pending_count"] for r in by_app),
        "failed_count": sum(r["failed_count"] for r in by_app),
        "refunded_count": sum(r["refunded_count"] for r in by_app),
    }
    settled = (
        totals["success_count"]
        + totals["failed_count"]
        + totals["refunded_count"]
        + sum(r["cancelled_count"] for r in by_app)
    )
    totals["success_rate"] = (
        round((totals["success_count"] + totals["refunded_count"]) / settled, 4)
        if settled
        else None
    )

    # Dense daily series — every day in the window present, so the chart has no
    # implicit gaps and the x-axis is evenly spaced.
    known = sorted(buckets.keys())
    series: Dict[str, Dict[str, float]] = {
        (start.date() + timedelta(days=i)).isoformat(): {a: 0.0 for a in known}
        for i in range(days)
    }
    for row in _rpc(sb, "chapa_daily_collections", params):
        day = str(row["day"])[:10]
        if day in series:
            series[day][row["app_id"]] = float(row["collected_etb"])

    daily = [
        {
            "day": day,
            "total_etb": round(sum(values.values()), 2),
            "by_app": {a: round(v, 2) for a, v in values.items()},
        }
        for day, values in sorted(series.items())
    ]

    return {
        "window": {
            "days": days,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "timezone": LEDGER_TIMEZONE,
        },
        "apps": known,
        "totals": totals,
        "by_app": by_app,
        "daily": daily,
    }


@router.get("/transactions")
async def list_transactions(
    days: int = Query(30, ge=1, le=365),
    app_id: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
):
    if app_id and app_id not in APP_IDS:
        raise HTTPException(400, f"Unknown app_id: {app_id}")
    if status and status not in STATUSES:
        raise HTTPException(400, f"Unknown status: {status}")

    sb = require_supabase()
    start, end = _window(days)
    offset = (page - 1) * per_page

    query = (
        sb.table(TABLE)
        .select(LIST_COLUMNS)
        .gte("created_at", start.isoformat())
        .lt("created_at", end.isoformat())
    )
    if app_id:
        query = query.eq("app_id", app_id)
    if status:
        query = query.eq("status", status)
    if search:
        # Commas, parens and asterisks are PostgREST filter syntax; strip them
        # so a search term cannot restructure the or_ expression.
        safe = re.sub(r"[,()*]", "", search).strip()[:100]
        if safe:
            query = query.or_(
                f"tx_ref.ilike.*{safe}*,external_user_id.ilike.*{safe}*"
            )

    # Fetch one extra row to learn whether a next page exists (range is
    # inclusive), matching the pattern already used by /api/gallery/images.
    res = (
        query.order("created_at", desc=True)
        .range(offset, offset + per_page)
        .execute()
    )
    rows = res.data or []

    return {
        "transactions": rows[:per_page],
        "page": page,
        "per_page": per_page,
        "has_next": len(rows) > per_page,
        "has_prev": page > 1,
    }


@router.post("/transactions/{tx_ref}/reverify")
async def reverify_transaction(tx_ref: str):
    """Ask Chapa again about a transaction stuck in pending."""
    return await reverify(require_supabase(), require_chapa(), tx_ref)


@router.post("/transactions/{tx_ref}/refund")
async def refund_from_dashboard(tx_ref: str, body: RefundBody = RefundBody()):
    """Refund from the dashboard. Admin scope: any business."""
    return await refund_transaction(
        require_supabase(), require_chapa(), tx_ref, amount_etb=body.amount_etb
    )
