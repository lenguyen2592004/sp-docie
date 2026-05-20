#!/usr/bin/env python3
"""Test import script to verify all dependencies."""

import sys

def test_import(module_name, package_name=None):
    """Test if module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

print("Testing imports...")
print("-" * 50)

success = True
success &= test_import('torch', 'PyTorch')
success &= test_import('transformers', 'Transformers')
success &= test_import('dgl', 'DGL')
success &= test_import('ot', 'POT')
success &= test_import('peft', 'PEFT')
success &= test_import('accelerate', 'Accelerate')
success &= test_import('bitsandbytes', 'BitsAndBytes')
success &= test_import('sklearn', 'Scikit-learn')
success &= test_import('wandb', 'WandB')
success &= test_import('lightning', 'Lightning')

print("-" * 50)
if success:
    print("All imports successful!")
    sys.exit(0)
else:
    print("Some imports failed. Install missing packages.")
    sys.exit(1)
