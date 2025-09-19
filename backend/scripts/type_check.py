#!/usr/bin/env python
"""
Type checking script for the DataSmith backend.

This script runs mypy on the Django project to check for type errors.
Usage: python scripts/type_check.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run_mypy():
    """Run mypy type checking on the project."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Modules to check
    modules_to_check = [
        "app/",
        "userauth/", 
        "datasource/",
        "core/",
    ]
    
    print("Running mypy type checking...")
    
    for module in modules_to_check:
        print(f"Checking {module}...")
        try:
            result = subprocess.run(
                ["python", "-m", "mypy", module], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode == 0:
                print(f"✓ {module} passed type checking")
            else:
                print(f"✗ {module} has type errors:")
                print(result.stdout)
                if result.stderr:
                    print("Errors:", result.stderr)
                    
        except subprocess.CalledProcessError as e:
            print(f"Error running mypy on {module}: {e}")
            return False
    
    print("Type checking complete!")
    return True


if __name__ == "__main__":
    success = run_mypy()
    sys.exit(0 if success else 1)