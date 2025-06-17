import sys
import subprocess
import time
import os
import tkinter as tk
from tkinter import messagebox


full_install_libs = [
    "pyside6==6.8.2.1"
]

priority_libs = {
    "cp311": {
        "GPU": [
            "https://download.pytorch.org/whl/cu126/torch-2.6.0%2Bcu126-cp311-cp311-win_amd64.whl#sha256=5ddca43b81c64df8ce0c59260566e648ee46b2622ab6a718e38dea3c0ca059a1",
            "nvidia-cuda-runtime-cu12==12.6.77",
            "nvidia-cublas-cu12==12.6.4.1",
            "nvidia-cuda-nvrtc-cu12==12.6.77",
            "nvidia-cuda-nvcc-cu12==12.6.77",
            "nvidia-cudnn-cu12==9.5.1.17",
        ],
        "CPU": [
            "https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=24c9d3d13b9ea769dd7bd5c11cfa1fc463fd7391397156565484565ca685d908"
        ]
    },
    "cp312": {
        "GPU": [
            "https://download.pytorch.org/whl/cu126/torch-2.6.0%2Bcu126-cp312-cp312-win_amd64.whl#sha256=b10c39c83e5d1afd639b5c9f5683b351e97e41390a93f59c59187004a9949924",
            "nvidia-cuda-runtime-cu12==12.6.77",
            "nvidia-cublas-cu12==12.6.4.1",
            "nvidia-cuda-nvrtc-cu12==12.6.77",
            "nvidia-cuda-nvcc-cu12==12.6.77",
            "nvidia-cudnn-cu12==9.5.1.17",

        ],
        "CPU": [
            "https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-win_amd64.whl#sha256=4027d982eb2781c93825ab9527f17fbbb12dbabf422298e4b954be60016f87d8"
        ]
    }
}

libs = [
    "av==14.1.0",
    "certifi==2025.01.31",
    "cffi==1.17.1",
    "charset-normalizer==3.4.1",
    "colorama==0.4.6",
    "coloredlogs==15.0.1",
    "ctranslate2==4.5.0",
    "faster-whisper==1.1.1",
    "filelock==3.17.0",
    "flatbuffers==25.2.10",
    "fsspec==2025.2.0",
    "huggingface-hub==0.28.1",
    "humanfriendly==10.0",
    "idna==3.10",
    "mpmath==1.3.0",
    "nltk==3.9.1",
    "numpy==1.26.4",
    "onnxruntime==1.20.1",
    "packaging==24.2",
    "protobuf==5.29.3",
    "psutil==6.1.1",
    "pycparser==2.22",
    "pyinstaller==6.11.1",
    "pyreadline3==3.5.4",
    "PyYAML==6.0.2",
    "regex==2024.11.6",
    "requests==2.32.3",
    "sounddevice==0.5.1",
    "sympy==1.13.1",
    "tokenizers==0.21.0",
    "tqdm==4.67.1",
    "typing_extensions==4.12.2",
    "urllib3==2.3.0",
]


start_time = time.time()

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
    if major == 3 and minor in [11, 12]:
        return tkinter_message_box("Confirmation", f"Python version {sys.version.split()[0]} was detected, which is compatible.\n\nClick YES to proceed or NO to exit.", yes_no=True)
    else:
        tkinter_message_box("Python Version Error", "This program requires Python 3.11 or 3.12\n\nPython versions prior to 3.11 or after 3.12 are not supported.\n\nExiting the installer...", type="error")
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

def install_libraries_with_retry(libraries, with_deps=False, max_retries=5, delay=3):
    failed_installations = []
    multiple_attempts = []
    
    for library in libraries:
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {library}")
                if with_deps:
                    command = ["uv", "pip", "install", library]
                else:
                    command = ["uv", "pip", "install", library, "--no-deps"]
                    
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=480)
                print(f"\033[92mSuccessfully installed {library}\033[0m")
                if attempt > 0:
                    multiple_attempts.append((library, attempt + 1))
                break
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    failed_installations.append(library)
    
    return failed_installations, multiple_attempts

def main():
    if not check_python_version_and_confirm():
        sys.exit(1)

    nvidia_gpu_detected = has_nvidia_gpu()
    message = "An NVIDIA GPU has been detected.\n\nDo you want to proceed with the installation?" if nvidia_gpu_detected else \
              "No NVIDIA GPU has been detected. CPU version will be installed.\n\nDo you want to proceed?"
    
    if not tkinter_message_box("Hardware Detection", message, yes_no=True):
        sys.exit(1)

    # 1. Install uv
    print("\033[92mInstalling uv:\033[0m")
    subprocess.run(["pip", "install", "uv"], check=True)

    # 2. Upgrade core packages
    print("\033[92mUpgrading pip, setuptools, and wheel:\033[0m")
    upgrade_pip_setuptools_wheel()

    # 3. Install priority libraries based on hardware
    print("\033[92mInstalling priority libraries:\033[0m")
    try:
        current_priority_libs = priority_libs[python_version][hardware_type]
        priority_failed, priority_multiple = install_libraries_with_retry(current_priority_libs)
    except KeyError:
        tkinter_message_box("Version Error", f"No libraries configured for Python {python_version} with {hardware_type} configuration", type="error")
        sys.exit(1)

    # 4. Install regular libraries
    print("\033[92mInstalling other libraries:\033[0m")
    other_failed, other_multiple = install_libraries_with_retry(libs)

    # 5. Install full installation libraries with dependencies
    print("\033[92mInstalling libraries with dependencies:\033[0m")
    full_failed, full_multiple = install_libraries_with_retry(full_install_libs, with_deps=True)

    # 6. Installation summary
    all_failed = priority_failed + other_failed + full_failed
    all_multiple = priority_multiple + other_multiple + full_multiple

    print("\n----- Installation Summary -----")
    if all_failed:
        print("\033[91m\nThe following libraries failed to install:\033[0m")
        for lib in all_failed:
            print(f"\033[91m- {lib}\033[0m")

    if all_multiple:
        print("\033[93m\nThe following libraries required multiple attempts:\033[0m")
        for lib, attempts in all_multiple:
            print(f"\033[93m- {lib} (took {attempts} attempts)\033[0m")

    if not all_failed and not all_multiple:
        print("\033[92mAll libraries installed successfully on the first attempt.\033[0m")
    elif not all_failed:
        print("\033[92mAll libraries were eventually installed successfully.\033[0m")

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")

if __name__ == "__main__":
    main()