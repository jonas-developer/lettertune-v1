## Set usage limits for safety
import os
import psycopg
from psycopg.rows import dict_row

MAX_GENERATIONS = int(os.getenv("MAX_GENERATIONS", "10"))

DDL = """
CREATE TABLE IF NOT EXISTS generation_counter (
  id INTEGER PRIMARY KEY,
  total BIGINT NOT NULL
);

INSERT INTO generation_counter (id, total)
VALUES (1, 0)
ON CONFLICT (id) DO NOTHING;
"""

def _get_conn():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set. Add a Render Postgres DB and set DATABASE_URL.")
    return psycopg.connect(db_url, autocommit=True)

def init_counter():
    # Call once on startup
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)

def get_usage():
    with _get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT total FROM generation_counter WHERE id=1;")
            row = cur.fetchone()
            return int(row["total"]) if row else 0

def check_and_increment():
    """
    Atomically:
    - If total >= MAX_GENERATIONS => reject
    - Else increment and return new total
    Safe under concurrent requests.
    """
    with _get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE generation_counter
                SET total = total + 1
                WHERE id = 1 AND total < %s
                RETURNING total;
                """,
                (MAX_GENERATIONS,),
            )
            row = cur.fetchone()
            if not row:
                # limit reached
                return False, get_usage()
            return True, int(row["total"])
