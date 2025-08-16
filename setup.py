#!/usr/bin/env python3
"""
Setup script for GPS Acquisition and Signal Generation System
Creates virtual environment and installs dependencies
"""

import os
import sys
import subprocess
import venv

def create_virtual_environment():
    """Create virtual environment"""
    venv_name = "gps_env"
    
    print("Creating virtual environment...")
    if os.path.exists(venv_name):
        print(f"Virtual environment '{venv_name}' already exists.")
        return venv_name
    
    venv.create(venv_name, with_pip=True)
    print(f"Virtual environment '{venv_name}' created successfully.")
    return venv_name

def install_requirements(venv_name):
    """Install requirements in virtual environment"""
    print("Installing requirements...")
    
    # Determine pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = os.path.join(venv_name, "Scripts", "pip")
    else:  # Unix/Linux/Mac
        pip_path = os.path.join(venv_name, "bin", "pip")
    
    try:
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    
    return True

def main():
    print("GPS Acquisition and Signal Generation System Setup")
    print("=" * 50)
    
    # Create virtual environment
    venv_name = create_virtual_environment()
    
    # Install requirements
    if install_requirements(venv_name):
        print("\n" + "=" * 50)
        print("Setup completed successfully!")
        print("=" * 50)
        print(f"\nTo activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print(f"  {venv_name}\\Scripts\\activate")
        else:  # Unix/Linux/Mac
            print(f"  source {venv_name}/bin/activate")
        
        print(f"\nTo run the test:")
        print(f"  python test_gps.py")
        
        print(f"\nTo test with real data:")
        print(f"  # After activating virtual environment")
        print(f"  # Load your real GPS data and use the acquisition module")
    else:
        print("Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
