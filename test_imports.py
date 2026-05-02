#!/usr/bin/env python
"""Quick test to verify all imports work correctly."""

try:
    print("Testing imports...")
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    print("✓ langchain_huggingface imported successfully")
    
    import torch
    print(f"✓ PyTorch imported successfully (version: {torch.__version__})")
    
    from transformers import AutoTokenizer
    print("✓ transformers imported successfully")
    
    print("\nAll imports successful! ✓")
    print("Your environment is properly configured.")
    print("\nYou can now run the locla_hf.py script.")
    print("Note: The first run will download the TinyLlama model (~6GB)")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
