import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
dsn = os.getenv("PG_DSN")

try:
    conn = psycopg2.connect(dsn)
    print("✅ Conexiune reușită la baza de date!")
except Exception as e:
    print("❌ Eroare:", e)