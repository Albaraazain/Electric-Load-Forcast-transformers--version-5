"""
CUDA utility functions for environment verification
"""

import torch
import sys
from typing import Dict, Union

def verify_cuda_environment() -> Dict[str, Union[bool, str]]:
    """
    Verify CUDA environment and return system information

    Returns:
        Dict containing CUDA environment information
    """
    env_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__,
        'python_version': sys.version,
    }

    if env_info['cuda_available']:
        env_info.update({
            'gpu_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'memory_allocated': f"{torch.cuda.memory_allocated(0)/1024**3:.2f} GB",
            'memory_cached': f"{torch.cuda.memory_reserved(0)/1024**3:.2f} GB",
        })

    return env_info

def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device based on CUDA availability

    Args:
        use_cuda: Whether to use CUDA if available

    Returns:
        torch.device: Device to use for computations
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        # Print device info
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device

def print_cuda_info():
    """Print detailed CUDA environment information"""
    env_info = verify_cuda_environment()

    print("\nCUDA Environment Information:")
    print("-" * 30)
    print(f"PyTorch Version: {env_info['pytorch_version']}")
    print(f"Python Version: {env_info['python_version']}")
    print(f"CUDA Available: {env_info['cuda_available']}")

    if env_info['cuda_available']:
        print(f"CUDA Version: {env_info['cuda_version']}")
        print(f"GPU Device: {env_info['gpu_name']}")
        print(f"Number of GPUs: {env_info['gpu_count']}")
        print(f"Current Device: {env_info['current_device']}")
        print(f"Memory Allocated: {env_info['memory_allocated']}")
        print(f"Memory Cached: {env_info['memory_cached']}")

        # Additional memory info
        print("\nMemory Summary:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved(i)/1024**3:.2f} GB")