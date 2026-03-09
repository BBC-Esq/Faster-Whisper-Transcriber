import sounddevice as sd
from typing import Optional, Tuple
from core.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_SAMPLE_RATE = 44100
REQUIRED_CHANNELS = 1
REQUIRED_DTYPE = "int16"


def get_default_input_device_id() -> Optional[int]:
    try:
        default_input, _ = sd.default.device
        return default_input
    except (sd.PortAudioError, OSError, TypeError) as e:
        logger.warning(f"Failed to get default input device: {e}")
        return None


def get_input_devices() -> list[dict]:
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        logger.warning(f"Failed to query audio devices: {e}")
        return []

    results = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            hostapi_name = hostapis[dev["hostapi"]]["name"]
            results.append({
                "id": idx,
                "name": dev["name"],
                "hostapi": hostapi_name,
                "max_input_channels": dev["max_input_channels"],
                "default_samplerate": int(dev["default_samplerate"]),
            })
    return results


def find_device_id_by_name(name: str, hostapi: str = "") -> Optional[int]:
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:
        return None

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and dev["name"] == name:
            if hostapi:
                dev_hostapi = hostapis[dev["hostapi"]]["name"]
                if dev_hostapi == hostapi:
                    return idx
            else:
                return idx

    if hostapi:
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and dev["name"] == name:
                return idx

    return None


def check_channel_support(device_id: int, channels: int, samplerate: int) -> bool:
    try:
        sd.check_input_settings(device=device_id, samplerate=samplerate, channels=channels)
        return True
    except (sd.PortAudioError, ValueError):
        return False


def get_supported_sample_rates(device_id: int, channels: int = 1) -> list[int]:
    sample_rates = [8000, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000]
    supported = []
    for rate in sample_rates:
        try:
            sd.check_input_settings(device=device_id, samplerate=rate, channels=channels)
            supported.append(rate)
        except (sd.PortAudioError, ValueError):
            continue
    return supported


def get_optimal_audio_settings(device_id: Optional[int] = None) -> Tuple[int, int, str]:
    if device_id is None:
        device_id = get_default_input_device_id()

    if device_id is None:
        logger.info("No input device found, using required settings")
        return REQUIRED_SAMPLE_RATE, REQUIRED_CHANNELS, REQUIRED_DTYPE

    try:
        device_info = sd.query_devices(device_id)
        device_name = device_info.get('name', 'Unknown')
        max_channels = device_info.get('max_input_channels', 0)
        default_samplerate = int(device_info.get('default_samplerate', 44100))
    except (sd.PortAudioError, OSError) as e:
        logger.warning(f"Failed to query device info: {e}")
        return REQUIRED_SAMPLE_RATE, REQUIRED_CHANNELS, REQUIRED_DTYPE

    if max_channels < 1:
        logger.warning(f"Device '{device_name}' reports no input channels")
        return REQUIRED_SAMPLE_RATE, REQUIRED_CHANNELS, REQUIRED_DTYPE

    if check_channel_support(device_id, REQUIRED_CHANNELS, REQUIRED_SAMPLE_RATE):
        logger.info(f"Device '{device_name}' supports {REQUIRED_SAMPLE_RATE} Hz mono - optimal settings")
        return REQUIRED_SAMPLE_RATE, REQUIRED_CHANNELS, REQUIRED_DTYPE

    if check_channel_support(device_id, 2, REQUIRED_SAMPLE_RATE):
        logger.info(f"Device '{device_name}' supports {REQUIRED_SAMPLE_RATE} Hz stereo - faster-whisper will convert to mono")
        return REQUIRED_SAMPLE_RATE, 2, REQUIRED_DTYPE

    for channels in [REQUIRED_CHANNELS, 2]:
        supported_rates = get_supported_sample_rates(device_id, channels)
        if supported_rates:
            if default_samplerate in supported_rates:
                selected_rate = default_samplerate
            else:
                selected_rate = max(supported_rates)
            logger.info(f"Device '{device_name}' using {selected_rate} Hz, {channels} ch - faster-whisper will convert")
            return selected_rate, channels, REQUIRED_DTYPE

    logger.warning(f"Could not determine optimal settings for '{device_name}', using defaults")
    return default_samplerate, min(max_channels, 2), REQUIRED_DTYPE
