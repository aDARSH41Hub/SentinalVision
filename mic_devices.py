"""List available audio input/output devices.

Utility script to enumerate microphones and speakers on the system.
Useful for determining which device index to use in audio recording.

Usage:
    python mic_devices.py
"""

import logging
from typing import List, Dict, Any

import sounddevice as sd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_devices() -> List[Dict[str, Any]]:
    """Query available audio devices.
    
    Returns:
        List of device info dictionaries
    """
    try:
        devices = sd.query_devices()
        logger.info("Found %d audio device(s)", len(devices))
        return devices
    except Exception as e:
        logger.error("Failed to query devices: %s", e, exc_info=True)
        raise RuntimeError(f"Could not query audio devices: {e}") from e


def print_devices(devices: List[Dict[str, Any]]) -> None:
    """Print device information in formatted table.
    
    Args:
        devices: List of device info dictionaries
    """
    print("\n" + "=" * 80)
    print("Available Audio Devices")
    print("=" * 80 + "\n")
    
    for i, device in enumerate(devices):
        print(f"[{i}] {device.get('name', 'Unknown Device')}")
        print(f"    Input Channels:  {device.get('max_input_channels', 0)}")
        print(f"    Output Channels: {device.get('max_output_channels', 0)}")
        print(f"    Sample Rate:     {device.get('default_samplerate', 0):.0f} Hz")
        print()
    
    print("=" * 80)
    print("\nTip: Use the device index [N] when creating AudioDetector or recording.")
    print("For example: sd.rec(..., device=0) uses device [0]")
    print()


def main() -> None:
    """Run device enumeration."""
    devices = get_devices()
    print_devices(devices)


if __name__ == "__main__":
    try:
        logger.info("Querying audio devices")
        main()
        logger.info("Device query completed successfully")
    except Exception as e:
        logger.error("Failed: %s", e, exc_info=True)
        exit(1)