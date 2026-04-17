#!/usr/bin/env python3
"""
Test script for the Ultimate AI Trading Bot
Validates all components are working correctly
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all components
from ai.ultimate_trading_bot import UltimateTradingBot
from ai.rl_trainer import TD3Trainer, TradingEnvironment
from ai.data_integrator import UltimateDataIntegrator
from ai.risk_manager import UltimateRiskManager

def test_imports():
    """Test all imports work"""
    print("🔍 Testing imports...")

    try:
        # Imports are now at module level
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_integrator():
    """Test data integration"""
    print("📊 Testing data integrator...")

    try:
        integrator = UltimateDataIntegrator()

        # Test with a simple symbol (this will work even without API keys)
        test_data = integrator._get_price_data('SPY')

        if isinstance(test_data, pd.DataFrame):
            print("✅ Data integrator working (basic functionality)")
            return True
        else:
            print("⚠️ Data integrator returned unexpected format")
            return False

    except Exception as e:
        print(f"❌ Data integrator error: {e}")
        return False

def test_rl_components():
    """Test RL components"""
    print("🤖 Testing RL components...")

    try:
        # Test environment creation
        env = TradingEnvironment(['MES', 'MNQ'], max_steps=10)

        # Test reset
        state = env.reset()
        assert isinstance(state, np.ndarray), "State should be numpy array"
        assert len(state) > 0, "State should not be empty"

        # Test step
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        assert isinstance(next_state, np.ndarray), "Next state should be numpy array"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(done, bool), "Done should be boolean"

        print("✅ RL environment working")
        return True

    except Exception as e:
        print(f"❌ RL components error: {e}")
        return False

def test_risk_manager():
    """Test risk management"""
    print("🛡️ Testing risk manager...")

    try:
        risk_mgr = UltimateRiskManager()

        # Test basic functionality
        can_trade, reason = risk_mgr.can_open_position('MES', 1000, 4000)
        assert isinstance(can_trade, bool), "Should return boolean"
        assert isinstance(reason, str), "Should return reason string"

        # Test position sizing
        size = risk_mgr.calculate_optimal_position_size('MES', 4000, 3980, 0.7)
        assert isinstance(size, (int, float)), "Should return numeric size"

        print("✅ Risk manager working")
        return True

    except Exception as e:
        print(f"❌ Risk manager error: {e}")
        return False

def test_pytorch_setup():
    """Test PyTorch is working"""
    print("🔥 Testing PyTorch setup...")

    try:
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y

        assert z.shape == (3, 3), "Tensor addition should work"

        # Test CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x_cuda = x.to(device)
            assert x_cuda.is_cuda, "Should be on CUDA"
            print("✅ PyTorch with CUDA working")
        else:
            print("✅ PyTorch (CPU) working")

        return True

    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("⚙️ Testing configuration...")

    try:
        from ai.ultimate_trading_bot import UltimateBotConfig

        # Test config values
        assert UltimateBotConfig.INITIAL_BALANCE > 0, "Initial balance should be positive"
        assert len(UltimateBotConfig.FUTURES_SYMBOLS) > 0, "Should have futures symbols"
        assert sum(UltimateBotConfig.ENSEMBLE_WEIGHTS.values()) > 0.99, "Weights should sum to ~1"

        print("✅ Configuration working")
        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("🧪 ULTIMATE AI TRADING BOT - COMPONENT TESTS")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("PyTorch Setup", test_pytorch_setup),
        ("Configuration", test_configuration),
        ("Data Integrator", test_data_integrator),
        ("RL Components", test_rl_components),
        ("Risk Manager", test_risk_manager),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The Ultimate Bot is ready to launch!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

def main():
    """Main test function"""
    success = run_all_tests()

    if success:
        print("\n🚀 Ready to launch the greatest AI trading bot ever!")
        print("Run: python launch_ultimate_bot.py --mode paper")
    else:
        print("\n🛠️ Please fix the failed tests before launching.")
        sys.exit(1)

if __name__ == "__main__":
    main()