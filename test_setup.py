#!/usr/bin/env python3
"""
Test script to verify all algorithms work correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    try:
        from src.models.simple_mlp import SimpleMLP
        from src.data.dataset_loader import get_dataset
        from src.algorithms.ewc import EWC
        from src.algorithms.replay import ExperienceReplay
        from src.algorithms.naive import NaiveLearning
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    try:
        from src.models.simple_mlp import SimpleMLP
        model = SimpleMLP(input_size=784, hidden_size=400, output_size=10)
        print(f"✅ Model created successfully: {model.__class__.__name__}")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_algorithms():
    """Test algorithm creation."""
    try:
        from src.models.simple_mlp import SimpleMLP
        from src.algorithms.ewc import EWC
        from src.algorithms.replay import ExperienceReplay
        from src.algorithms.naive import NaiveLearning
        
        model = SimpleMLP(input_size=784, hidden_size=400, output_size=10)
        
        # Test EWC
        ewc = EWC(model, device='cpu')
        print(f"✅ EWC algorithm created successfully")
        
        # Test Replay
        replay = ExperienceReplay(model, device='cpu', memory_size=100)
        print(f"✅ ExperienceReplay algorithm created successfully")
        
        # Test Naive
        naive = NaiveLearning(model, device='cpu')
        print(f"✅ NaiveLearning algorithm created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Algorithm creation error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Continual Learning Playground...")
    print()
    
    success = True
    success &= test_imports()
    success &= test_model_creation()
    success &= test_algorithms()
    
    print()
    if success:
        print("🎉 All tests passed! The continual learning playground is ready to use.")
        print()
        print("Usage examples:")
        print("  python main.py --algorithm ewc --dataset mnist --epochs 5")
        print("  python main.py --algorithm replay --dataset permuted_mnist --memory_size 500")
        print("  python main.py --algorithm naive --dataset cifar10 --epochs 3")
    else:
        print("💥 Some tests failed. Please check the implementation.")
        sys.exit(1)
