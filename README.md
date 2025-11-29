# 🎙️ Faster Whisper Transcriber

> Voice-to-text transcription powered by CTranslate2 — speak, transcribe, and paste into your favorite LLM in seconds.

![Screenshot](https://github.com/user-attachments/assets/2e5b613b-6a14-4274-9fc1-2cf8a6cde9cc)

## ✨ Features

- 🎤 **Record & Transcribe** — Capture audio directly from your microphone
- 📁 **File Transcription** — Transcribe existing audio files
- 🌍 **Translation** — Translate foreign language audio to English
- 📋 **Auto-Clipboard** — Transcriptions are automatically copied and ready to paste
- ⚡ **GPU Accelerated** — CUDA support for fast transcription

## 📋 Requirements

| Requirement | Notes |
|-------------|-------|
| [Python 3.11](https://www.python.org/downloads/release/python-3119/) or [3.12](https://www.python.org/downloads/release/python-31210/) | Required |
| [Git](https://git-scm.com/downloads) | Required |
| [git-lfs](https://git-lfs.com/) | Required |
| Windows | Linux/macOS support welcome — testers needed! |

## 🚀 Installation
```bash
# 1. Download and extract the latest release, then navigate to the folder containing main.py

# 2. Create and activate virtual environment
python -m venv .
.\Scripts\activate

# 3. Install dependencies
python install.py
```

## 💡 Usage
```bash
# Activate environment and start
.\Scripts\activate
python main.py
```

1. **Select a model** and click "Update Settings" (first use will download the model automatically)
2. **Record** your voice or **select an audio file** to transcribe
3. **Paste** anywhere with `Ctrl+V` — the transcription is already in your clipboard!

> **Note:** The "Copy to Clipboard" button in the sidebar lets you edit and re-copy the transcription if needed.

## 🤝 Contributing

Interested in helping bring this to Linux or macOS? Feel free to open an issue or PR!
