import sys
import subprocess
import time
import os
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

def build_library_list():
    all_libs = list(libs)

    if hardware_type == "GPU":
        if python_version not in gpu_libs:
            tkinter_message_box("Version Error", f"No GPU libraries configured for Python {python_version}", type="error")
            sys.exit(1)
        all_libs = gpu_libs[python_version] + all_libs

    return all_libs

def install_libraries(libraries, max_retries=5, delay=3):
    command = ["uv", "pip", "install"] + libraries

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

    all_libs = build_library_list()
    print(f"\033[92mInstalling {len(all_libs)} libraries ({hardware_type} configuration):\033[0m")
    success, attempts = install_libraries(all_libs)

    print("\n----- Installation Summary -----")
    if not success:
        print(f"\033[91mInstallation failed after {attempts} attempts.\033[0m")
    elif attempts > 1:
        print(f"\033[93mAll libraries installed successfully after {attempts} attempts.\033[0m")
    else:
        print("\033[92mAll libraries installed successfully on the first attempt.\033[0m")

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")

if __name__ == "__main__":
    main()