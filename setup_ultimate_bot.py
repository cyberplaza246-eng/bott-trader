#!/usr/bin/env python3
"""
Setup script for the Ultimate AI Trading Bot
Installs all required dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True,
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("🚀 ULTIMATE AI TRADING BOT - SETUP")
    print("="*60)

    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor} detected")

    # Update pip
    if not run_command("python -m pip install --upgrade pip", "Updating pip"):
        sys.exit(1)

    # Install core dependencies
    if not run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
                      "Installing PyTorch (CPU version)"):
        print("⚠️ PyTorch installation failed, trying alternative...")
        run_command("pip install torch==2.0.0+cpu torchvision==0.15.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",
                   "Installing PyTorch alternative")

    # Install other dependencies
    requirements_files = [
        'requirements_ultimate.txt',
        'requirements.txt'  # Fallback to original requirements
    ]

    for req_file in requirements_files:
        if os.path.exists(req_file):
            if run_command(f"pip install -r {req_file}", f"Installing from {req_file}"):
                break
        else:
            print(f"⚠️ {req_file} not found, trying next...")

    # Install additional packages that might be missing
    additional_packages = [
        'numpy',
        'pandas',
        'requests',
        'gym',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy'
    ]

    run_command(f"pip install {' '.join(additional_packages)}", "Installing additional packages")

    # Create necessary directories
    directories = [
        'logs',
        'models',
        'data',
        'results'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

    # Create .env template if it doesn't exist
    if not os.path.exists('.env'):
        env_template = """# Ultimate AI Trading Bot Configuration

# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
POLYMARKET_API_KEY=your_polymarket_key_here
TWITTER_API_KEY=your_twitter_key_here
REDDIT_API_KEY=your_reddit_key_here

# Trading Configuration
INITIAL_BALANCE=50000
TRADING_MODE=paper
RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05

# Broker Configuration (choose one)
# Interactive Brokers
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1

# Alpaca
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# TD Ameritrade
TD_ACCOUNT_ID=your_td_account
TD_REFRESH_TOKEN=your_td_refresh_token
"""
        with open('.env', 'w') as f:
            f.write(env_template)
        print("📄 Created .env template file")

    # Test installation
    print("\n🧪 Testing installation...")
    try:
        import torch
        print("✅ PyTorch installed successfully")
    except ImportError:
        print("⚠️ PyTorch not available - some features may not work")

    try:
        import numpy as np
        import pandas as pd
        print("✅ Core data science libraries installed")
    except ImportError:
        print("❌ Core libraries missing")

    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python test_ultimate_bot.py")
    print("3. Start paper trading: python launch_ultimate_bot.py --mode paper")
    print("\n🚀 Ready to launch the greatest AI trading bot ever!")

if __name__ == "__main__":
    main()