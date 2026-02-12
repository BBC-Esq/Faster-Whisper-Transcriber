#!/usr/bin/env python
"""
Build script for FWTranscriber
Compiles the project with PyInstaller and prepares the release package
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import time

# ANSI color codes for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")

def get_folder_size(folder_path):
    """Calculate folder size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

def get_file_size(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'PyInstaller', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        print_success(f"PyInstaller {version} is installed")
        return True
    except subprocess.CalledProcessError:
        print_error("PyInstaller is not installed")
        print_info("Install it with: pip install pyinstaller")
        return False

def clean_build_folders():
    """Clean build and dist folders"""
    print_info("Cleaning build folders...")
    
    folders_to_clean = ['build', 'dist']
    for folder in folders_to_clean:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print_success(f"Removed {folder}/")
    
    # Also remove .spec cache
    spec_file = Path(__file__).parent / 'FWTranscriber.spec'
    if spec_file.exists():
        print_info(f"Using spec file: {spec_file}")

def run_pyinstaller():
    """Run PyInstaller compilation"""
    print_info("Starting PyInstaller compilation...")
    print_warning("This may take 2-5 minutes...")
    
    spec_file = 'FWTranscriber.spec'
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'PyInstaller', spec_file, '--clean'],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print_success(f"Compilation completed in {minutes}m {seconds}s")
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"PyInstaller compilation failed with exit code {e.returncode}")
        return False

def verify_exe():
    """Verify that the exe file was created"""
    exe_path = Path('dist') / 'FWTranscriber.exe'
    
    if not exe_path.exists():
        print_error(f"Executable not found at {exe_path}")
        return False
    
    size_mb = get_file_size(exe_path)
    print_success(f"Executable created: {exe_path} ({size_mb:.1f} MB)")
    return True

def prepare_release():
    """Prepare the release folder with exe and external resources"""
    print_info("Preparing release package...")
    
    release_dir = Path('FWTranscriber_Release')
    
    # Remove old release folder if exists
    if release_dir.exists():
        print_info(f"Removing old release folder: {release_dir}")
        shutil.rmtree(release_dir)
    
    # Create release folder
    release_dir.mkdir(exist_ok=True)
    print_success(f"Created release folder: {release_dir}")
    
    total_size = 0
    
    # Copy exe file
    exe_src = Path('dist') / 'FWTranscriber.exe'
    exe_dst = release_dir / 'FWTranscriber.exe'
    
    if exe_src.exists():
        shutil.copy2(exe_src, exe_dst)
        size = get_file_size(exe_dst)
        total_size += size
        print_success(f"Copied FWTranscriber.exe ({size:.1f} MB)")
    else:
        print_error("FWTranscriber.exe not found in dist/")
        return False
    
    # Copy cuda_lib folder (if exists)
    cuda_lib_src = Path('cuda_lib')
    cuda_lib_dst = release_dir / 'cuda_lib'
    
    if cuda_lib_src.exists() and cuda_lib_src.is_dir():
        shutil.copytree(cuda_lib_src, cuda_lib_dst)
        size = get_folder_size(cuda_lib_dst)
        total_size += size
        print_success(f"Copied cuda_lib/ ({size:.1f} MB)")
    else:
        print_warning("cuda_lib/ not found - CPU only version")
    
    # Copy models folder (if exists)
    models_src = Path('models')
    models_dst = release_dir / 'models'
    
    if models_src.exists() and models_src.is_dir():
        # Check if there are any model folders (not just README.md)
        model_folders = [f for f in models_src.iterdir() if f.is_dir()]
        
        if model_folders:
            shutil.copytree(models_src, models_dst)
            size = get_folder_size(models_dst)
            total_size += size
            print_success(f"Copied models/ with {len(model_folders)} model(s) ({size:.1f} MB)")
        else:
            print_warning("models/ is empty - models will be downloaded on first use")
    else:
        print_warning("models/ not found - models will be downloaded on first use")
    
    # Copy config.yaml (if exists)
    config_src = Path('config.yaml')
    config_dst = release_dir / 'config.yaml'
    
    if config_src.exists():
        shutil.copy2(config_src, config_dst)
        print_success("Copied config.yaml")
    else:
        print_warning("config.yaml not found - will be created on first run")
    
    # Create README.txt in release folder
    readme_content = """FWTranscriber - Faster Whisper Transcriber

QUICK START:
1. Run FWTranscriber.exe
2. Select audio input device
3. Click "Start Recording" or "Transcribe File"

FOLDER STRUCTURE:
- FWTranscriber.exe    - Main application
- cuda_lib/            - CUDA libraries (for GPU acceleration)
- models/              - Whisper models (downloaded automatically if not present)
- config.yaml          - Configuration file (created on first run)
- logs/                - Application logs (created on first run)

REQUIREMENTS:
- Windows 10 or later
- For GPU: NVIDIA GPU with CUDA support

For more information, visit:
https://github.com/BBC-Esq/Faster-Whisper-Transcriber
"""
    
    readme_path = release_dir / 'README.txt'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print_success("Created README.txt")
    
    print_success(f"\nTotal release package size: {total_size:.1f} MB")
    
    return True

def create_zip_archive():
    """Create a ZIP archive of the release package"""
    print_info("Creating ZIP archive...")
    
    release_dir = Path('FWTranscriber_Release')
    if not release_dir.exists():
        print_error("Release folder not found")
        return False
    
    zip_name = 'FWTranscriber_Release.zip'
    
    # Remove old zip if exists
    if os.path.exists(zip_name):
        os.remove(zip_name)
    
    try:
        shutil.make_archive('FWTranscriber_Release', 'zip', release_dir)
        size = get_file_size(zip_name)
        print_success(f"Created {zip_name} ({size:.1f} MB)")
        return True
    except Exception as e:
        print_error(f"Failed to create ZIP archive: {e}")
        return False

def print_summary(success):
    """Print build summary"""
    print_header("BUILD SUMMARY")
    
    if success:
        print_success("Build completed successfully!")
        print()
        print_info("Release package location:")
        print(f"  {Colors.BOLD}FWTranscriber_Release/{Colors.RESET}")
        print()
        print_info("To test the build:")
        print(f"  {Colors.BOLD}cd FWTranscriber_Release{Colors.RESET}")
        print(f"  {Colors.BOLD}.\\FWTranscriber.exe{Colors.RESET}")
        print()
        print_info("Check logs after running:")
        print(f"  {Colors.BOLD}FWTranscriber_Release\\logs\\transcriber_YYYYMMDD.log{Colors.RESET}")
        print()
        
        # List what was included
        release_dir = Path('FWTranscriber_Release')
        print_info("Package contents:")
        
        if (release_dir / 'FWTranscriber.exe').exists():
            size = get_file_size(release_dir / 'FWTranscriber.exe')
            print(f"  ✓ FWTranscriber.exe ({size:.1f} MB)")
        
        if (release_dir / 'cuda_lib').exists():
            size = get_folder_size(release_dir / 'cuda_lib')
            print(f"  ✓ cuda_lib/ ({size:.1f} MB) - GPU support enabled")
        else:
            print(f"  ✗ cuda_lib/ - CPU only")
        
        if (release_dir / 'models').exists():
            size = get_folder_size(release_dir / 'models')
            model_count = len([f for f in (release_dir / 'models').iterdir() if f.is_dir()])
            print(f"  ✓ models/ ({size:.1f} MB) - {model_count} model(s)")
        else:
            print(f"  ⚠ models/ - will download on first use")
        
        if (release_dir / 'config.yaml').exists():
            print(f"  ✓ config.yaml")
        else:
            print(f"  ⚠ config.yaml - will be created on first run")
        
        print()
    else:
        print_error("Build failed!")
        print_info("Check the error messages above for details")

def main():
    """Main build function"""
    print_header("FWTranscriber Build Script")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print_info(f"Working directory: {script_dir}")
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_info(f"Python version: {py_version}")
    
    # Step 1: Check PyInstaller
    print_header("Step 1: Check Dependencies")
    if not check_pyinstaller():
        print_error("Please install PyInstaller first: pip install pyinstaller")
        sys.exit(1)
    
    # Step 2: Clean old build files
    print_header("Step 2: Clean Build Folders")
    clean_build_folders()
    
    # Step 3: Run PyInstaller
    print_header("Step 3: Compile with PyInstaller")
    if not run_pyinstaller():
        print_summary(False)
        sys.exit(1)
    
    # Step 4: Verify exe
    print_header("Step 4: Verify Executable")
    if not verify_exe():
        print_summary(False)
        sys.exit(1)
    
    # Step 5: Prepare release package
    print_header("Step 5: Prepare Release Package")
    if not prepare_release():
        print_summary(False)
        sys.exit(1)
    
    # Step 6: Create ZIP archive (optional)
    # print_header("Step 6: Create ZIP Archive (Optional)")
    # create_zip_archive()
    
    # Print summary
    print_summary(True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\nBuild cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
