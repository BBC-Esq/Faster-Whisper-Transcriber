from __future__ import annotations

class TranscriberError(Exception):
    pass

class ModelLoadError(TranscriberError):
    pass

class AudioRecordingError(TranscriberError):
    pass

class AudioSaveError(TranscriberError):
    pass

class TranscriptionError(TranscriberError):
    pass

class ConfigurationError(TranscriberError):
    pass