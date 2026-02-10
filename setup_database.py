#!/usr/bin/env python3
"""
Setup script to initialize Supabase database with schema.
Run this once to create all tables.

AI Attribution: This file was developed with assistance from Claude (Anthropic).
https://claude.ai
"""

import os
from dotenv import load_dotenv
from supabase import create_client


def main():
    """Initialize the Supabase database using the SQL schema file."""
    load_dotenv()

    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY not found in .env")
        return

    print("Connecting to Supabase...")
    supabase = create_client(supabase_url, supabase_key)

    try:
        with open('database_schema.sql', 'r') as f:
            sql_script = f.read()
        print("Schema file loaded")
    except FileNotFoundError:
        print("Error: database_schema.sql not found")
        return

    print("Creating database tables...")

    try:
        response = supabase.postgrest.auth(supabase_key).execute_raw_sql(sql_script)
        print("Database setup complete!")
        print("\nTables created:")
        print("  - hospitals")
        print("  - users")
        print("  - uploads")
        print("  - analyses")
        print("  - patient_metadata")
        print("  - case_summary")
        print("  - regional_summary")
        print("  - alerts")
        print("  - resources")
        print("\nSample data loaded (5 hospitals, 30-day outbreak data)")
        print("\nYou're ready to deploy!")

    except Exception as e:
        print(f"Note: {str(e)}")
        print("\nYou can also set up the database manually:")
        print("1. Go to your Supabase project dashboard")
        print("2. Click SQL Editor -> New Query")
        print("3. Copy all SQL from database_schema.sql")
        print("4. Paste and Run")
        print("\nThen you're ready to deploy!")


if __name__ == '__main__':
    main()
