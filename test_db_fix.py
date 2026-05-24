#!/usr/bin/env python3
"""
Test database connection with the updated sanitization logic.
"""

import sys
import os

# Add app to path
sys.path.insert(0, '.')

from app.utils.db import PG_DSN, sanitize_neon_dsn, get_conn

print("="*70)
print("DATABASE CONNECTION TEST")
print("="*70)

print(f"\n[1] Original DSN loaded from .env:")
print(f"    {PG_DSN}")

sanitized = sanitize_neon_dsn(PG_DSN)
print(f"\n[2] Sanitized DSN (search_path removed if present):")
print(f"    {sanitized}")

print(f"\n[3] Attempting connection...")
try:
    conn = get_conn()
    print(f"    ✅ Connection successful!")
    
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"    PostgreSQL version: {version[:50]}...")
    
    conn.close()
    print("\n✅ ALL TESTS PASSED - Database connection is working!")
    
except Exception as e:
    print(f"    ❌ Connection failed!")
    print(f"    Error: {type(e).__name__}: {str(e)[:200]}")
    print("\n🔧 Troubleshooting:")
    print("    1. Make sure PostgreSQL is running")
    print("    2. Check if the DSN in .env is correct")
    print("    3. For Neon: Use the unpooled connection (direct endpoint)")
    print("       Pooled connections do NOT support search_path parameter")
