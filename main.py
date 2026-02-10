"""
MediAlert - Pneumonia Detection & Pandemic Surveillance Platform
Main Entry Point

This script provides a unified interface to run the application.

AI Attribution: Code structure generated with assistance from Claude AI (Anthropic)
via Cursor IDE on February 2026.

Usage:
    python main.py              # Run web application (default)
    python main.py --setup      # Initialize database and environment
"""

import argparse
import sys
import os

def run_setup():
    """Initialize database and project environment."""
    print("Setting up MediAlert...")
    print("Step 1: Creating directories...")
    
    dirs = ['data/raw', 'data/processed', 'data/outputs', 'models', 'notebooks', 'uploads']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Step 2: Setting up database...")
    os.system("python setup_database.py")
    
    print("\nSetup complete!")
    print("Next: Run 'python main.py' to start the application")

def run_web_app():
    """Launch the MediAlert web application."""
    print("Starting MediAlert Web Application...")
    print("Open http://localhost:5000 in your browser")
    print("\nMake sure you have configured .env with required credentials.")
    print("See .env.example for template.\n")
    
    from app.main import app
    app.run(debug=True, port=5000, host='0.0.0.0')

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='MediAlert - Pneumonia Detection & Pandemic Surveillance Platform'
    )
    parser.add_argument(
        '--setup', 
        action='store_true',
        help='Initialize database and environment before first run'
    )
    
    args = parser.parse_args()
    
    if args.setup:
        run_setup()
    else:
        run_web_app()

if __name__ == "__main__":
    main()
