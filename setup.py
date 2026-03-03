"""
MediAlert Project Setup Script

This script prepares the project environment:
- Creates necessary directories
- Checks dependencies
- Initializes database
- Validates configuration

via Cursor IDE on February 2026.

Usage:
    python setup.py
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create project directory structure."""
    dirs = [
        'data/raw',
        'data/processed',
        'data/outputs',
        'models',
        'notebooks',
        'scripts',
        'uploads'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("Created directory structure")

def check_dependencies():
    """Check if required packages are installed."""
    required = ['flask', 'requests']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("All core dependencies are installed")
    return True

def check_env():
    """Check for .env configuration."""
    if not os.path.exists('.env'):
        print("Warning: .env file not found")
        print("Copy .env.example to .env and fill in your credentials")
        return False
    print("Found .env configuration")
    return True

def run_pipeline():
    """Run data ingestion, feature building, and model training."""
    print("\n--- Running Pipeline Scripts ---")
    scripts = [
        ("make_dataset.py", []),
        ("build_features.py", []),
        ("model.py", ["--train"])
    ]
    for script, args in scripts:
        script_path = Path('scripts') / script
        if script_path.exists():
            print(f"> Executing {script}...")
            # We use check=False so it doesn't hard-fail if data is missing during setup
            subprocess.run([sys.executable, str(script_path)] + args, check=False)
        else:
            print(f"> Warning: {script} not found in scripts/")
    print("--- Pipeline Scripts Completed ---\n")

def main():
    print("=" * 60)
    print("MediAlert Project Setup")
    print("=" * 60)
    
    create_directories()
    deps_ok = check_dependencies()
    env_ok = check_env()
    
    if not deps_ok:
        sys.exit(1)
    
    if not env_ok:
        print("\nConfiguration incomplete. Please create .env file.")
        sys.exit(1)
    
    run_pipeline()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run the application: python main.py")
    print("\n")

if __name__ == "__main__":
    main()
