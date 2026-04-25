<div align="center">

<img width="621" height="367" alt="image" src="https://github.com/user-attachments/assets/7fa15938-5c18-424b-8f2b-83b30380e523" />

### Reliably transcribe your voice or an audio file on CPU or GPU!

<img width="1090" height="220" alt="image" src="https://github.com/user-attachments/assets/76908cd1-954c-4752-94a4-6423ec610b1e" />

<br>
</div>

## Features

- Voice recording with real-time waveform visualization
- Single file and batch (multi-file) transcription
- Recursive directory scanning for batch processing
- Configurable file type filtering
- Multiple output formats: txt, srt, vtt, tsv, json
- Output to clipboard, source directory, or custom directory
- Optional speaker labeling mode for recorded calls and imported audio files
- Speaker label settings for 2-8 expected speakers
- Optional voice enrollment for the primary speaker, used only when the match is confident
- Optional NVIDIA Parakeet TDT 0.6B v3 transcription through ONNX Runtime
- Real-time system monitoring (CPU, RAM, GPU, VRAM, Power)
- Global hotkey support
- Dockable clipboard and file transcription panels
- All settings persisted between sessions

## Supported Models

Uses the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) library, which provides CTranslate2-based inference for OpenAI's Whisper models. Supports both transcription and translation tasks depending on the model selected.

This fork also adds optional local Parakeet support:

- `parakeet-tdt-0.6b-v3-onnx`
- Runs locally through [onnx-asr](https://pypi.org/project/onnx-asr/) and ONNX Runtime
- Uses `CUDAExecutionProvider` when ONNX Runtime GPU is available
- Avoids NVIDIA NeMo as an application dependency
- Downloads and caches the ONNX model on first use

Parakeet is transcription-only. Whisper models remain available and are still the recommended fallback for compatibility.

## Speaker Labels

The clipboard panel includes a `Speaker Labels` checkbox. When enabled, the app assigns speaker names to both live recordings and imported audio files.

In Settings:

- `Speaker Labels...` lets you set 2-8 expected speaker names, such as Lawyer, Client, Adjuster, or Interpreter.
- `Voice Enrollment...` records a short sample of the primary speaker. The sample is used only when the match is distinct enough; otherwise the app falls back to normal speaker ordering.

Speaker labeling uses fast local audio features and does not send audio to a cloud service. It is intended as a lightweight helper, not a guaranteed forensic diarization system.

## Windows Installer

> Download and run [```FasterWhisperTranscriber_Setup.exe```](https://github.com/BBC-Esq/Faster-Whisper-Transcriber/releases/latest/download/FasterWhisperTranscriber_Setup.exe).

## Install And Run from Virtual Environment

> Download the latest release...unzip and extract...go to the directory containing ```main.py```...run these commands in order:

```
python -m venv .
```

```
.\Scripts\activate
```

```
python install.py
```

```
python main.py
```

During install, the app can optionally install Parakeet ONNX support. The regular Faster Whisper install is unaffected if the optional Parakeet install is skipped or fails.
