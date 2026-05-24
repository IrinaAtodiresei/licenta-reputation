#!/usr/bin/env python3
"""
Test database connection.
Run this script to verify PostgreSQL connection works.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test PostgreSQL connection."""
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL:
        print("❌ ERROR: DATABASE_URL not found in .env file")
        return False
    
    print(f"📌 Connection string: {DATABASE_URL}")
    print()
    
    try:
        import psycopg
        print("✓ psycopg2 imported successfully")
        
        # Try to connect
        print("🔄 Connecting to database...")
        conn = psycopg.connect(DATABASE_URL)
        print("✅ Connected successfully!")
        
        # Try to execute a simple query
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"📊 PostgreSQL version: {version}")
        
        # Try to check if views exist
        print()
        print("🔍 Checking for reputation schema views...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'reputation' 
                AND table_type = 'VIEW'
                ORDER BY table_name;
            """)
            views = cur.fetchall()
            
            if views:
                print(f"✅ Found {len(views)} views in 'reputation' schema:")
                for (view_name,) in views:
                    print(f"   • {view_name}")
            else:
                print("⚠️  No views found in 'reputation' schema.")
        
        # Try to check tables in reputation schema
        print()
        print("🔍 Checking for tables in reputation schema...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'reputation' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            tables = cur.fetchall()
            
            if tables:
                print(f"✅ Found {len(tables)} tables in 'reputation' schema:")
                for (table_name,) in tables:
                    print(f"   • {table_name}")
            else:
                print("⚠️  No tables found in 'reputation' schema.")
        
        # Try to check tables in public schema
        print()
        print("🔍 Checking for tables in public schema...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            tables = cur.fetchall()
            
            if tables:
                print(f"✅ Found {len(tables)} tables in 'public' schema:")
                for (table_name,) in tables:
                    print(f"   • {table_name}")
            else:
                print("⚠️  No tables found in 'public' schema.")
        
        conn.close()
        print()
        print("✅ ALL TESTS PASSED - Database connection is working!")
        return True
        
    except ImportError as e:
        print(f"❌ ERROR: Import failed: {e}")
        print("Install psycopg with: pip install psycopg")
        return False
        
    except Exception as e:
        print(f"❌ ERROR: Connection failed")
        print(f"   Details: {type(e).__name__}: {e}")
        print()
        print("🔧 Troubleshooting tips:")
        print("   1. Check if PostgreSQL is running: sudo service postgresql status")
        print("   2. Verify DATABASE_URL in .env file")
        print("   3. Check PostgreSQL credentials (username/password)")
        print("   4. Check if database 'neondb' exists")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
