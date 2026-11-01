-- Dashboard aggregates for the Chapa ledger.
--
-- These exist because PostgREST caps a plain select at 1000 rows: aggregating
-- in Python would mean paginating the whole ledger on every dashboard load and
-- would silently under-report once volume passes the cap. Doing the GROUP BY in
-- Postgres keeps the totals correct at any size.
--
-- Both functions are `security invoker` (the default) on purpose. The ledger has
-- RLS enabled with no policies, so these return nothing to anon/authenticated
-- callers even if the grants below were ever loosened — only the service role,
-- which bypasses RLS, sees rows.

-- Per-business, per-status rollup over a window. The caller pivots this.
create or replace function public.chapa_app_summary(
  p_from timestamptz,
  p_to   timestamptz
)
returns table (
  app_id    text,
  status    text,
  txn_count bigint,
  total_etb numeric
)
language sql
stable
as $$
  select
    t.app_id,
    t.status,
    count(*)::bigint                 as txn_count,
    coalesce(sum(t.amount_etb), 0)   as total_etb
  from public.chapa_transactions t
  where t.created_at >= p_from
    and t.created_at <  p_to
  group by t.app_id, t.status;
$$;

-- Net collections per local day, per business. Powers the dashboard chart.
--
-- Bucketed in Ethiopian local time, not UTC, so a 9pm EAT payment lands on the
-- day the business actually made it. Counts only status='success', so a refund
-- removes the money from the day it was taken — the chart reads as "money we
-- still hold", which is what `succeeded_at` anchoring implies.
create or replace function public.chapa_daily_collections(
  p_from timestamptz,
  p_to   timestamptz
)
returns table (
  day           date,
  app_id        text,
  collected_etb numeric
)
language sql
stable
as $$
  select
    (t.succeeded_at at time zone 'Africa/Addis_Ababa')::date as day,
    t.app_id,
    coalesce(sum(t.amount_etb), 0)                           as collected_etb
  from public.chapa_transactions t
  where t.status = 'success'
    and t.succeeded_at is not null
    and t.succeeded_at >= p_from
    and t.succeeded_at <  p_to
  group by 1, 2
  order by 1, 2;
$$;

-- Nothing but the service role should be able to execute these.
revoke all on function public.chapa_app_summary(timestamptz, timestamptz)
  from public, anon, authenticated;
revoke all on function public.chapa_daily_collections(timestamptz, timestamptz)
  from public, anon, authenticated;

grant execute on function public.chapa_app_summary(timestamptz, timestamptz)
  to service_role;
grant execute on function public.chapa_daily_collections(timestamptz, timestamptz)
  to service_role;

-- PostgREST caches the schema; without this the new functions 404 from the
-- Supabase client until the cache expires on its own.
notify pgrst, 'reload schema';
