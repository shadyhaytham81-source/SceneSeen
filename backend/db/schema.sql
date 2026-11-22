-- SceneSeen Supabase schema (planned for milestone 3; the backend uses an in-memory store until then).
-- Mirrors backend/schemas.py and the SRS class diagram (3.9). Run in the Supabase SQL editor.
create extension if not exists vector;

create type user_role as enum ('viewer', 'brand_partner', 'stylist', 'admin');
create type tag_source as enum ('user', 'suggested', 'none');
create type verified_status as enum ('pending', 'approved');

create table brands (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  price_tier text
);

-- profile row per Supabase Auth user (auth.users holds email and password hash)
create table users (
  id uuid primary key references auth.users on delete cascade,
  name text,
  role user_role not null default 'viewer',
  brand_id uuid references brands,            -- brand partners only
  preferences jsonb not null default '{}',
  instagram text                               -- stylists only
);

create table shows (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  type text not null default 'series'
);

create table episodes (
  id uuid primary key default gen_random_uuid(),
  show_id uuid not null references shows on delete cascade,
  season int not null,
  number int not null,
  unique (show_id, season, number)
);

create table stylist_shows (
  stylist_id uuid references users on delete cascade,
  show_id uuid references shows on delete cascade,
  primary key (stylist_id, show_id)
);

create table products (
  id uuid primary key default gen_random_uuid(),
  brand_id uuid not null references brands on delete cascade,
  title text not null,
  category text not null,
  price numeric(10, 2),                        -- EGP
  product_url text not null,
  image_url text,
  colour text,
  image_embedding vector(512)
);
create index on products (category);
create index on products using hnsw (image_embedding vector_cosine_ops);

create table scans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references users on delete set null,   -- scanning needs no account
  input_kind text not null,                            -- camera | screenshot
  episode_id uuid references episodes,
  tag_source tag_source not null default 'none',
  flagged boolean not null default false,
  frame_embedding vector(512),
  processing_ms int,
  created_at timestamptz not null default now()
);
create index on scans using hnsw (frame_embedding vector_cosine_ops);

create table detected_items (
  id uuid primary key default gen_random_uuid(),
  scan_id uuid not null references scans on delete cascade,
  category text not null,
  colour text,
  confidence real not null,
  box jsonb not null
);

create table match_results (
  id uuid primary key default gen_random_uuid(),
  detected_item_id uuid not null references detected_items on delete cascade,
  product_id uuid not null references products on delete cascade,
  rank int not null,
  similarity real not null,
  label text not null check (label in ('Exact', 'Similar'))
);

create table verified_items (
  id uuid primary key default gen_random_uuid(),
  episode_id uuid not null references episodes on delete cascade,
  product_id uuid not null references products on delete cascade,
  character text,
  scene_time text,
  confirmed_by uuid not null references users,
  approved_by uuid references users,
  status verified_status not null default 'pending'
);

create table clicks (
  id uuid primary key default gen_random_uuid(),
  match_id uuid references match_results on delete set null,
  product_id uuid not null references products on delete cascade,
  user_id uuid references users on delete set null,
  utm_source text not null default 'sceneseen',
  clicked_at timestamptz not null default now()
);

create table saved_items (
  user_id uuid references users on delete cascade,
  product_id uuid references products on delete cascade,
  saved_at timestamptz not null default now(),
  primary key (user_id, product_id)
);
