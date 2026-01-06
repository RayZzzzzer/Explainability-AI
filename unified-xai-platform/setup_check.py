"""
Setup script for the Unified XAI Platform
Run this script to verify installation and setup
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - TOO OLD")
        print("   Please install Python 3.8 or higher")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'streamlit',
        'tensorflow',
        'numpy',
        'librosa',
        'lime',
        'shap',
        'cv2',
        'PIL',
        'matplotlib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_directory_structure():
    """Verify project directory structure"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        'utils',
        'xai_methods',
        'models',
        'data',
        'docs'
    ]
    
    missing = []
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists() and path.is_dir():
            print(f"✅ {dir_name}/ - exists")
        else:
            print(f"❌ {dir_name}/ - NOT FOUND")
            missing.append(dir_name)
    
    return len(missing) == 0, missing

def create_missing_directories(missing_dirs):
    """Create missing directories"""
    print("\nCreating missing directories...")
    for dir_name in missing_dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_name}/")

def check_files():
    """Check if required files exist"""
    print("\nChecking required files...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'utils/__init__.py',
        'utils/preprocessing.py',
        'utils/model_loader.py',
        'utils/compatibility.py',
        'xai_methods/__init__.py',
        'xai_methods/lime_explainer.py',
        'xai_methods/gradcam_explainer.py',
        'xai_methods/shap_explainer.py'
    ]
    
    missing = []
    for file_name in required_files:
        path = Path(file_name)
        if path.exists() and path.is_file():
            print(f"✅ {file_name} - exists")
        else:
            print(f"❌ {file_name} - NOT FOUND")
            missing.append(file_name)
    
    return len(missing) == 0, missing

def main():
    """Main setup verification"""
    print("="*60)
    print("UNIFIED XAI PLATFORM - SETUP VERIFICATION")
    print("="*60)
    
    all_ok = True
    
    # Check Python version
    if not check_python_version():
        all_ok = False
        print("\n⚠️ Please install Python 3.8 or higher and try again")
        return
    
    # Check directory structure
    dirs_ok, missing_dirs = check_directory_structure()
    if not dirs_ok:
        print("\n⚠️ Some directories are missing. Creating them...")
        create_missing_directories(missing_dirs)
    
    # Check files
    files_ok, missing_files = check_files()
    if not files_ok:
        print("\n⚠️ Some required files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        all_ok = False
    
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        print("\n⚠️ Some dependencies are missing. Installing...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("\n✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("\n❌ Failed to install dependencies")
            print("   Please run: pip install -r requirements.txt")
            all_ok = False
    
    # Final summary
    print("\n" + "="*60)
    if all_ok and deps_ok and files_ok:
        print("✅ SETUP COMPLETE - READY TO RUN!")
        print("="*60)
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
        print("\nThen open your browser to:")
        print("   http://localhost:8501")
    else:
        print("⚠️ SETUP INCOMPLETE - PLEASE FIX ISSUES ABOVE")
        print("="*60)
        if not deps_ok:
            print("\nInstall missing dependencies:")
            print("   pip install -r requirements.txt")
        if not files_ok:
            print("\nSome files are missing. Please ensure all project files are present.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
