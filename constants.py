full_install_libs = [
    "pyside6==6.8.2.1"
]

priority_libs = {
    "cp311": {
        "GPU": [
            "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-win_amd64.whl",
            "nvidia-cuda-runtime-cu12==12.8.90",
            "nvidia-cublas-cu12==12.8.4.1",
            "nvidia-cuda-nvrtc-cu12==12.8.93",
            "nvidia-cuda-nvcc-cu12==12.4.131",
            "nvidia-cudnn-cu12==9.10.2.21",
        ],
        "CPU": [
            "https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp311-cp311-win_amd64.whl"
        ]
    },
    "cp312": {
        "GPU": [
            "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl",
            "nvidia-cuda-runtime-cu12==12.8.90",
            "nvidia-cublas-cu12==12.8.4.1",
            "nvidia-cuda-nvrtc-cu12==12.8.93",
            "nvidia-cuda-nvcc-cu12==12.4.131",
            "nvidia-cudnn-cu12==9.10.2.21",
        ],
        "CPU": [
            "https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp312-cp312-win_amd64.whl"
        ]
    }
}

libs = [
    "av==15.1.0",
    "certifi==2025.8.3",
    "cffi==2.0.0",
    "charset-normalizer==3.4.3",
    "colorama==0.4.6",
    "coloredlogs==15.0.1",
    "ctranslate2==4.6.0",
    "faster-whisper==1.1.1",
    "filelock==3.19.1",
    "flatbuffers==25.2.10",
    "fsspec==2025.3.0",
    "huggingface-hub==0.34.4",
    "humanfriendly==10.0",
    "idna==3.10",
    "mpmath==1.3.0",
    "nltk==3.9.1",
    "numpy==2.2.6",
    "onnxruntime==1.20.1",
    "packaging==25.0",
    "protobuf==6.32.1",
    "psutil==7.0.0",
    "pycparser==2.23",
    "pyinstaller==6.11.1",
    "pyreadline3==3.5.4",
    "PyYAML==6.0.2",
    "regex==2025.9.1",
    "requests==2.32.5",
    "sounddevice==0.5.2",
    "sympy==1.13.3",
    "tokenizers==0.22.0",
    "tqdm==4.67.1",
    "typing_extensions==4.15.0",
    "urllib3==2.5.0",
]