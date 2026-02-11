#!/usr/bin/env python3
"""
Utility for pre-downloading Whisper models.

Usage:
    python download_models.py --all              # Download all models
    python download_models.py --model large-v3   # Download specific model
    python download_models.py --list             # List local models
"""

import argparse
import sys

from core.models.download import (
    download_model_to_local,
    list_local_models,
    get_models_directory,
)
from core.models.loader import _make_repo_string, _extract_model_name_from_repo
from core.models.metadata import ModelMetadata


def download_all_models():
    """Download all popular models."""
    models_to_download = [
        ("tiny", "float32"),
        ("base", "float32"),
        ("small", "float32"),
        ("medium", "float32"),
        ("large-v3", "float16"),
        ("large-v3-turbo", "float16"),
        ("distil-whisper-large-v3", "bfloat16"),
    ]
    
    print("Downloading all models...")
    success_count = 0
    fail_count = 0
    
    for model_name, quant in models_to_download:
        repo_id = _make_repo_string(model_name, quant)
        local_name = _extract_model_name_from_repo(repo_id)
        
        try:
            print(f"\n--- Downloading {model_name} ({quant}) ---")
            download_model_to_local(repo_id, local_name)
            print(f"✓ Downloaded {model_name} ({quant})")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
            fail_count += 1
    
    print(f"\n--- Summary ---")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {fail_count}")


def download_specific_model(model_name: str, quant: str = "float32"):
    """Download a specific model.
    
    Args:
        model_name: Model name (e.g., "large-v3")
        quant: Quantization type (e.g., "float32", "bfloat16")
    """
    repo_id = _make_repo_string(model_name, quant)
    local_name = _extract_model_name_from_repo(repo_id)
    
    print(f"Downloading {model_name} ({quant})...")
    
    try:
        model_path = download_model_to_local(repo_id, local_name)
        print(f"✓ Downloaded {model_name} ({quant}) to {model_path}")
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        sys.exit(1)


def show_local_models():
    """Show list of locally available models."""
    models_dir = get_models_directory()
    print(f"Models directory: {models_dir}")
    print()
    
    models = list_local_models()
    if not models:
        print("No local models found.")
        print()
        print("To download models, use:")
        print("  python download_models.py --model large-v3-turbo")
        print("  python download_models.py --all")
    else:
        print(f"Local models ({len(models)}):")
        for model in sorted(models):
            print(f"  ✓ {model}")


def show_available_models():
    """Show list of all available models to download."""
    print("Available models:")
    print()
    
    all_models = ModelMetadata.get_all_model_names()
    for model_name in all_models:
        info = ModelMetadata.get_model_info(model_name)
        if info:
            translation = "✓ Translation" if info.supports_translation else "✗ No translation"
            print(f"  - {model_name:30s} {translation}")
    
    print()
    print("Common quantization types: float32, float16, bfloat16")
    print()
    print("Example usage:")
    print("  python download_models.py --model large-v3 --quant bfloat16")


def main():
    parser = argparse.ArgumentParser(
        description="Download and manage Whisper models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py --list
  python download_models.py --model large-v3-turbo
  python download_models.py --model large-v3 --quant bfloat16
  python download_models.py --all
  python download_models.py --available
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all popular models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Download specific model (e.g., large-v3)"
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="float32",
        help="Quantization type (default: float32)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List locally available models"
    )
    parser.add_argument(
        "--available",
        action="store_true",
        help="Show all available models to download"
    )
    
    args = parser.parse_args()
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    # Execute commands
    if args.list:
        show_local_models()
    elif args.available:
        show_available_models()
    elif args.all:
        download_all_models()
    elif args.model:
        download_specific_model(args.model, args.quant)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
