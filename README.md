<div align="center">

<img width="621" height="367" alt="image" src="https://github.com/user-attachments/assets/7fa15938-5c18-424b-8f2b-83b30380e523" />

### Reliably transcribe your voice or an audio file on CPU or GPU!

<img width="659" height="190" alt="image" src="https://github.com/user-attachments/assets/093ea17b-6d9b-470c-9c95-b7334f7e93b2" />

<br>
</div>

## Features

- Voice recording with real-time waveform visualization
- Single file and batch (multi-file) transcription
- Recursive directory scanning for batch processing
- Configurable file type filtering
- Multiple output formats: txt, srt, vtt, tsv, json
- Output to clipboard, source directory, or custom directory
- Real-time system monitoring (CPU, RAM, GPU, VRAM, Power)
- Global hotkey support
- Dockable clipboard and file transcription panels
- All settings persisted between sessions

## Supported Models

Uses the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) library, which provides CTranslate2-based inference for OpenAI's Whisper models.  Supports both transcription and translation tasks depending on the model selected.

## ⚙️ Windows Installer
> Download and run [```FasterWhisperTranscriber_Setup.exe```](https://github.com/BBC-Esq/Faster-Whisper-Transcriber/releases/latest/download/FasterWhisperTranscriber_Setup.exe).

## 💻 Install And Run from Virtual Environment
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
