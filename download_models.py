import os
import sys
import argparse
from huggingface_hub import snapshot_download

def get_default_model_dir():
    """Get the default model directory for ComfyUI"""
    # Try to find ComfyUI models directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "..", "ComfyUI", "models", "mmaudio"),  # If ComfyUI is next to current dir
        os.path.join(script_dir, "..", "..", "models", "mmaudio"),  # If we're in ComfyUI/custom_nodes
        os.path.join(os.path.expanduser("~"), "ComfyUI", "models", "mmaudio"),  # User's home directory
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.dirname(os.path.dirname(path))):  # Check if models dir exists
            return path
    return os.path.join(script_dir, "models", "mmaudio")  # Fallback to local models directory

def download_bigvgan_models(output_dir):
    """
    Download BigVGAN models from HuggingFace
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        nvidia_bigvgan_vocoder_path = os.path.join(output_dir, "nvidia", "bigvgan_v2_44khz_128band_512x")
        
        if not os.path.exists(nvidia_bigvgan_vocoder_path):
            print(f"Downloading nvidia bigvgan vocoder model to: {nvidia_bigvgan_vocoder_path}")
            snapshot_download(
                repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
                ignore_patterns=["*3m*",],
                local_dir=nvidia_bigvgan_vocoder_path,
                local_dir_use_symlinks=False,
            )
            print("\nDownload complete! The models are ready to use with ComfyUI.")
        else:
            print(f"Model already exists at {nvidia_bigvgan_vocoder_path}")
            
    except Exception as e:
        print(f"\nError downloading models: {str(e)}", file=sys.stderr)
        print("\nIf you're having connection issues, please check your internet connection and try again.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    default_dir = get_default_model_dir()
    
    parser = argparse.ArgumentParser(
        description="Download BigVGAN models for ComfyUI-MMAudio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Download to default location ({default_dir}):
  python download_models.py

  # Download to custom location:
  python download_models.py --output-dir /path/to/your/models/directory
"""
    )
    
    parser.add_argument("--output-dir", type=str, default=default_dir,
                      help=f"Directory where models should be downloaded (default: {default_dir})")
    
    args = parser.parse_args()
    print(f"\nDownloading models to: {args.output_dir}")
    download_bigvgan_models(args.output_dir)
