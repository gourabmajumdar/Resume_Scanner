import subprocess
import sys

def install_requirements():
    """Install required packages."""
    requirements = [
        "streamlit",
        "pandas", 
        "numpy",
        "plotly",
        "scikit-learn",
        "PyPDF2",
        "python-docx"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

if __name__ == "__main__":
    print("🚀 Setting up Resume Screening Tool...")
    install_requirements()
    print("\n✅ Setup complete!")
    print("\nTo run the app, use: streamlit run resume_screener.py")