import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

PG_DSN = os.getenv("PG_DSN")

if not PG_DSN:
    raise RuntimeError("PG_DSN lipsește din .env / secrets")

with psycopg.connect(PG_DSN, autocommit=True) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE reputation.mention
            ADD COLUMN IF NOT EXISTS subreddit TEXT,
            ADD COLUMN IF NOT EXISTS post_id TEXT,
            ADD COLUMN IF NOT EXISTS comment_id TEXT,
            ADD COLUMN IF NOT EXISTS reddit_url TEXT,
            ADD COLUMN IF NOT EXISTS raw_json JSONB;
        """)

print("GATA. Coloanele pentru Proof of source au fost adăugate.")