import sys
import subprocess
import time
import os
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

gpu_libs = {
    "cp311": [
        "nvidia-cuda-runtime-cu12==12.8.90",
        "nvidia-cublas-cu12==12.8.4.1",
        "nvidia-cudnn-cu12==9.10.2.21",
    ],
    "cp312": [
        "nvidia-cuda-runtime-cu12==12.8.90",
        "nvidia-cublas-cu12==12.8.4.1",
        "nvidia-cudnn-cu12==9.10.2.21",
    ],
    "cp313": [
        "nvidia-cuda-runtime-cu12==12.8.90",
        "nvidia-cublas-cu12==12.8.4.1",
        "nvidia-cudnn-cu12==9.10.2.21",
    ],
}

libs = [
    "ctranslate2==4.6.2",
    "faster-whisper==1.2.1",
    "nltk", # required by my program
    "psutil", # required by my program
    "pynput", # required by my program
    "pyside6", # required by my program
    "sounddevice", # required by my program
    "sympy==1.13.3", # set to known torch compatibility
]


start_time = time.time()

def enable_ansi_colors():
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        stdout_handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(stdout_handle, ctypes.byref(mode))
        mode.value |= 0x0004
        kernel32.SetConsoleMode(stdout_handle, mode)

def has_nvidia_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
hardware_type = "GPU" if has_nvidia_gpu() else "CPU"

def tkinter_message_box(title, message, type="info", yes_no=False):
    root = tk.Tk()
    root.withdraw()
    if yes_no:
        result = messagebox.askyesno(title, message)
    elif type == "error":
        messagebox.showerror(title, message)
        result = False
    else:
        messagebox.showinfo(title, message)
        result = True
    root.destroy()
    return result

def check_python_version_and_confirm():
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if major == 3 and minor in [11, 12, 13]:
        return tkinter_message_box("Confirmation", f"Python version {sys.version.split()[0]} was detected, which is compatible.\n\nClick YES to proceed or NO to exit.", yes_no=True)
    else:
        tkinter_message_box("Python Version Error", "This program requires Python 3.11, 3.12 or 3.13\n\nPython versions prior to 3.11 or after 3.13 are not supported.\n\nExiting the installer...", type="error")
        return False

def upgrade_pip_setuptools_wheel(max_retries=5, delay=3):
    upgrade_commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--no-cache-dir"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "setuptools", "--no-cache-dir"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "wheel", "--no-cache-dir"]
    ]

    for command in upgrade_commands:
        package = command[5]
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Upgrading {package}...")
                process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=480)
                print(f"\033[92mSuccessfully upgraded {package}\033[0m")
                break
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

def install_gpu_libraries_to_cuda_lib(gpu_packages, max_retries=5, delay=3):
    """Install GPU libraries to cuda_lib/ directory."""
    cuda_lib_path = Path(__file__).parent / "cuda_lib"
    cuda_lib_path.mkdir(exist_ok=True)
    
    print(f"\033[92mInstalling {len(gpu_packages)} GPU libraries to cuda_lib/:\033[0m")
    command = ["uv", "pip", "install", "--target", str(cuda_lib_path)] + gpu_packages
    
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}: Installing GPU libraries...")
            subprocess.run(command, check=True, text=True, timeout=1800)
            print(f"\033[92mSuccessfully installed GPU libraries to cuda_lib/\033[0m")
            return True, attempt + 1
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed.")
            if e.stderr:
                print(f"Error: {e.stderr.strip()}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
    
    return False, max_retries

def install_regular_libraries(packages, max_retries=5, delay=3):
    """Install regular libraries to standard site-packages."""
    print(f"\033[92mInstalling {len(packages)} regular libraries:\033[0m")
    command = ["uv", "pip", "install"] + packages

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {len(libraries)} libraries...")
            subprocess.run(command, check=True, text=True, timeout=1800)
            print(f"\033[92mSuccessfully installed all {len(libraries)} libraries\033[0m")
            return True, attempt + 1
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed.")
            if e.stderr:
                print(f"Error: {e.stderr.strip()}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    return False, max_retries

def offer_model_download():
    """Offer to download Whisper models after installation."""
    message = (
        "Installation complete!\n\n"
        "Would you like to download Whisper models now?\n\n"
        "Models will be downloaded to the 'models/' folder.\n"
        "You can also download them later using download_models.py\n\n"
        "Download models now?"
    )
    
    if tkinter_message_box("Model Download", message, yes_no=True):
        # Ask which models to download
        choice_message = (
            "Which models would you like to download?\n\n"
            "• Recommended: large-v3-turbo (fastest, good quality, ~800 MB)\n"
            "• All models: Download all available models (~10 GB)\n"
            "• Skip: Download later manually\n\n"
            "Download recommended model?"
        )
        
        if tkinter_message_box("Model Selection", choice_message, yes_no=True):
            print("\n\033[92mDownloading recommended model (large-v3-turbo)...\033[0m")
            try:
                result = subprocess.run(
                    [sys.executable, "download_models.py", "--model", "large-v3-turbo", "--quant", "bfloat16"],
                    check=True,
                    capture_output=False,
                    text=True,
                    timeout=3600  # 1 hour timeout for large downloads
                )
                print("\033[92mModel downloaded successfully!\033[0m")
            except subprocess.TimeoutExpired:
                print("\033[91mModel download timed out. Please download manually later.\033[0m")
            except subprocess.CalledProcessError:
                print("\033[91mModel download failed. You can download later using download_models.py\033[0m")
            except Exception as e:
                print(f"\033[91mModel download error: {e}\033[0m")
        else:
            print("\n\033[93mSkipping model download. You can download later using:\033[0m")
            print("  python download_models.py --model large-v3-turbo")
            print("  python download_models.py --list")

def main():
    enable_ansi_colors()

    if not check_python_version_and_confirm():
        sys.exit(1)

    nvidia_gpu_detected = has_nvidia_gpu()
    message = "An NVIDIA GPU has been detected.\n\nDo you want to proceed with the installation?" if nvidia_gpu_detected else \
              "No NVIDIA GPU has been detected. CPU version will be installed.\n\nDo you want to proceed?"

    if not tkinter_message_box("Hardware Detection", message, yes_no=True):
        sys.exit(1)

    print("\033[92mInstalling uv:\033[0m")
    subprocess.run(["pip", "install", "uv"], check=True)

    print("\033[92mUpgrading pip, setuptools, and wheel:\033[0m")
    upgrade_pip_setuptools_wheel()

    # Install GPU libraries to cuda_lib/ if GPU detected
    gpu_success = True
    gpu_attempts = 0
    if hardware_type == "GPU":
        if python_version not in gpu_libs:
            tkinter_message_box("Version Error", f"No GPU libraries configured for Python {python_version}", type="error")
            sys.exit(1)
        
        gpu_packages = gpu_libs[python_version]
        gpu_success, gpu_attempts = install_gpu_libraries_to_cuda_lib(gpu_packages)
        
        if not gpu_success:
            print(f"\033[91mGPU libraries installation failed after {gpu_attempts} attempts.\033[0m")
            sys.exit(1)

    # Install regular libraries to site-packages
    print(f"\n\033[92mInstalling regular libraries ({hardware_type} configuration):\033[0m")
    regular_success, regular_attempts = install_regular_libraries(libs)

    print("\n----- Installation Summary -----")
    
    if hardware_type == "GPU":
        if gpu_attempts > 1:
            print(f"\033[93mGPU libraries installed to cuda_lib/ after {gpu_attempts} attempts.\033[0m")
        else:
            print(f"\033[92mGPU libraries installed to cuda_lib/ on the first attempt.\033[0m")
    
    if not regular_success:
        print(f"\033[91mRegular libraries installation failed after {regular_attempts} attempts.\033[0m")
    elif regular_attempts > 1:
        print(f"\033[93mRegular libraries installed successfully after {regular_attempts} attempts.\033[0m")
    else:
        print("\033[92mAll libraries installed successfully on the first attempt.\033[0m")

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")
    
    # Offer to download models
    if regular_success:
        offer_model_download()

if __name__ == "__main__":
    main()