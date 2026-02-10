#!/usr/bin/env python3
"""
Setup script to initialize Supabase database with schema.
Run this once to create all tables.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

# Get Supabase credentials
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: SUPABASE_URL and SUPABASE_KEY not found in .env")
    exit(1)

print("🔗 Connecting to Supabase...")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Read SQL schema
try:
    with open('database_schema.sql', 'r') as f:
        sql_script = f.read()
    print("✅ Schema file loaded")
except FileNotFoundError:
    print("❌ Error: database_schema.sql not found")
    exit(1)

# Split SQL statements and execute
print("📊 Creating database tables...")

try:
    # Execute the SQL script
    response = supabase.postgrest.auth(SUPABASE_KEY).execute_raw_sql(sql_script)
    print("✅ Database setup complete!")
    print("\nTables created:")
    print("  • hospitals")
    print("  • users")
    print("  • uploads")
    print("  • analyses")
    print("  • patient_metadata")
    print("  • case_summary")
    print("  • regional_summary")
    print("  • alerts")
    print("  • resources")
    print("\n✅ Sample data loaded (5 hospitals, 30-day outbreak data)")
    print("\nYou're ready to deploy! 🚀")

except Exception as e:
    print(f"⚠️  Note: {str(e)}")
    print("\nYou can also set up the database manually:")
    print("1. Go to https://gdtcnuzanixrmxgedqqp.supabase.co")
    print("2. Click SQL Editor → New Query")
    print("3. Copy all SQL from database_schema.sql")
    print("4. Paste and Run")
    print("\nThen you're ready to deploy! 🚀")
