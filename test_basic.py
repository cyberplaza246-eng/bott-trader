#!/usr/bin/env python3
"""
Simple test script for the Ultimate AI Trading Bot
Tests basic functionality without heavy dependencies
"""

import sys
import os
import importlib.util

def test_file_exists(filepath):
    """Test if file exists"""
    if os.path.exists(filepath):
        print(f"✅ {filepath} exists")
        return True
    else:
        print(f"❌ {filepath} missing")
        return False

def test_import_module(module_name, filepath):
    """Test if module can be imported"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ {module_name} imports successfully")
        return True
    except Exception as e:
        print(f"❌ {module_name} import failed: {e}")
        return False

def test_basic_dependencies():
    """Test basic Python dependencies"""
    dependencies = [
        'os', 'sys', 'json', 'datetime', 'time',
        'collections', 'threading', 'logging'
    ]

    failed = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} missing")
            failed.append(dep)

    return len(failed) == 0

def test_numpy_pandas():
    """Test numpy and pandas"""
    try:
        import numpy as np
        import pandas as pd
        print("✅ NumPy and Pandas available")
        return True
    except ImportError as e:
        print(f"⚠️ NumPy/Pandas not available: {e}")
        print("   Run: pip install numpy pandas")
        return False

def test_requests():
    """Test requests library"""
    try:
        import requests
        print("✅ Requests library available")
        return True
    except ImportError:
        print("⚠️ Requests not available - API calls won't work")
        print("   Run: pip install requests")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("🧪 ULTIMATE AI TRADING BOT - BASIC TESTS")
    print("="*60)

    # Test file existence
    print("📁 Checking file structure...")
    files_to_check = [
        'src/ai/ultimate_trading_bot.py',
        'src/ai/rl_trainer.py',
        'src/ai/data_integrator.py',
        'src/ai/risk_manager.py',
        'launch_ultimate_bot.py',
        'requirements_ultimate.txt',
        'ULTIMATE_BOT_README.md'
    ]

    file_results = []
    for filepath in files_to_check:
        file_results.append(test_file_exists(filepath))

    # Test basic dependencies
    print("\n🐍 Testing basic Python dependencies...")
    basic_deps_ok = test_basic_dependencies()

    # Test data science libraries
    print("\n📊 Testing data science libraries...")
    data_libs_ok = test_numpy_pandas()

    # Test HTTP library
    print("\n🌐 Testing HTTP library...")
    requests_ok = test_requests()

    # Test module imports (syntax check)
    print("\n📦 Testing module imports...")
    modules_to_test = [
        ('ultimate_trading_bot', 'src/ai/ultimate_trading_bot.py'),
        ('rl_trainer', 'src/ai/rl_trainer.py'),
        ('data_integrator', 'src/ai/data_integrator.py'),
        ('risk_manager', 'src/ai/risk_manager.py'),
        ('launch_ultimate_bot', 'launch_ultimate_bot.py')
    ]

    import_results = []
    for module_name, filepath in modules_to_test:
        import_results.append(test_import_module(module_name, filepath))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)

    all_files_ok = all(file_results)
    all_imports_ok = all(import_results)

    print(f"Files present: {sum(file_results)}/{len(file_results)}")
    print(f"Modules import: {sum(import_results)}/{len(import_results)}")
    print(f"Basic deps: {'✅' if basic_deps_ok else '❌'}")
    print(f"Data libs: {'✅' if data_libs_ok else '❌'}")
    print(f"HTTP lib: {'✅' if requests_ok else '❌'}")

    overall_success = all_files_ok and all_imports_ok and basic_deps_ok

    if overall_success:
        print("\n🎉 Basic tests passed! The bot structure is sound.")
        print("Next: Install full dependencies with setup_ultimate_bot.py")
        return True
    else:
        print("\n⚠️ Some basic tests failed. Check the errors above.")
        return False

def main():
    """Main test function"""
    success = run_all_tests()

    if success:
        print("\n🚀 Ready for full setup!")
        print("Run: python setup_ultimate_bot.py")
    else:
        print("\n🛠️ Please fix the issues before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()