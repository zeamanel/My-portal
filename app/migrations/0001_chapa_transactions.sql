-- my-portal Chapa ledger.
-- One row per payment attempt. `app_id` identifies which app initiated it
-- (odaflux / shaleqa / ethiogram / behalf), and `external_user_id` is that
-- app's own user identifier — my-portal never resolves it, just echoes it back.

create table if not exists public.chapa_transactions (
  id              bigint generated always as identity primary key,
  tx_ref          text not null unique,       -- our reference, sent to Chapa
  app_id          text not null check (app_id in ('odaflux','shaleqa','ethiogram','behalf')),
  external_user_id text not null,             -- that app's user id, opaque here
  amount_etb      numeric(12,2) not null check (amount_etb > 0),
  currency        text not null default 'ETB',
  status          text not null default 'pending'
                  check (status in ('pending','success','failed','cancelled','refunded')),
  chapa_checkout_url text,
  chapa_ref       text,                       -- Chapa's own reference, from verify
  metadata        jsonb not null default '{}'::jsonb,
  return_url      text,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),
  succeeded_at    timestamptz,
  failed_at       timestamptz,
  refunded_at     timestamptz,
  failure_reason  text
);

create index if not exists chapa_transactions_app_status_idx
  on public.chapa_transactions (app_id, status, created_at desc);

create index if not exists chapa_transactions_external_user_idx
  on public.chapa_transactions (app_id, external_user_id);

create index if not exists chapa_transactions_created_idx
  on public.chapa_transactions (created_at desc);

-- Touch updated_at on every update.
create or replace function public.touch_chapa_transactions_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_chapa_transactions_updated_at on public.chapa_transactions;
create trigger trg_chapa_transactions_updated_at
  before update on public.chapa_transactions
  for each row execute function public.touch_chapa_transactions_updated_at();

-- Service role only. Nothing in any browser reads this table.
alter table public.chapa_transactions enable row level security;
-- No policies: service-role bypasses RLS; anon/authenticated get nothing.

comment on table public.chapa_transactions is
  'Per-attempt Chapa payment ledger. Only my-portal reads/writes it with the '
  'service-role key; calling apps poll by tx_ref to learn the outcome.';

-- PostgREST caches the schema; without this the new table 404s from the
-- Supabase client until the cache expires on its own.
notify pgrst, 'reload schema';
