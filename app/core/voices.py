"""
Defines supported TTS voices and validation.

- Stores list of allowed voice names
- Checks if given voice is valid and supported

This file ensures only approved voices are used in the system.
"""


SUPPORTED_VOICES = {"en-US-JennyNeural", "en-US-GuyNeural", "en-IN-NeerjaNeural",}


def is_supported_voice(voice: str) -> bool:
    return isinstance(voice, str) and voice.strip() in SUPPORTED_VOICES

