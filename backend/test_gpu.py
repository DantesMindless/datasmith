#!/usr/bin/env python3
"""
GPU Test Script for DataSmith
Tests CUDA availability and functionality in Docker container
"""

import torch
import sys

def test_cuda():
    """Test CUDA availability and basic functionality"""
    print("🚀 Testing GPU/CUDA support in Docker container")
    print("=" * 50)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU devices found: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({memory_total:.1f} GB)")
        
        # Test GPU computation
        try:
            print("\n🔬 Testing GPU computation...")
            device = torch.device('cuda:0')
            
            # Create test tensors
            a = torch.randn(1000, 1000).to(device)
            b = torch.randn(1000, 1000).to(device)
            
            # Perform computation
            c = torch.mm(a, b)
            print("✅ GPU matrix multiplication successful!")
            print(f"Result shape: {c.shape}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ GPU computation failed: {e}")
            return False
    else:
        print("❌ No CUDA support detected")
        
        # Check possible reasons
        print("\n🔍 Possible issues:")
        print("- NVIDIA drivers not installed on host")
        print("- Docker GPU runtime not configured")
        print("- Container doesn't have GPU access")
        
        return False

if __name__ == "__main__":
    success = test_cuda()
    sys.exit(0 if success else 1)