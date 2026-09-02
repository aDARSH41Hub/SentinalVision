"""Microphone audio level test.

Records a few seconds of audio from specified microphone and measures
signal levels (peak amplitude, RMS). Useful for debugging microphone
issues and verifying input levels before running detector.

Usage:
    python mic_test.py
    python mic_test.py --device 1 --duration 3
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import sounddevice as sd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
SAMPLE_RATE = 16000


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Test microphone signal levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mic_test.py                # Use device 1, 5 seconds
  python mic_test.py --device 0     # Use device 0 (default)
  python mic_test.py --duration 3   # Record for 3 seconds
        """,
    )
    
    parser.add_argument(
        "--device",
        type=int,
        default=1,
        help="Audio device index (default: 1)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Recording duration in seconds (default: 5)",
    )
    
    return parser.parse_args()


def record_audio(
    device: int,
    duration: int,
    sample_rate: int,
) -> np.ndarray:
    """Record audio from microphone.
    
    Args:
        device: Device index
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio array (mono, float32)
        
    Raises:
        RuntimeError: If recording fails
    """
    try:
        logger.info("Recording from device %d for %d seconds...", device, duration)
        print(f"\nRecording for {duration} seconds...")
        print("Speak clearly and fairly close to the microphone!\n")
        
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
        )
        
        sd.wait()
        audio = audio.squeeze()
        
        logger.info("Recording complete")
        return audio
        
    except Exception as e:
        logger.error("Recording failed: %s", e, exc_info=True)
        raise RuntimeError(f"Could not record audio: {e}") from e


def analyze_audio(audio: np.ndarray) -> Tuple[float, float]:
    """Compute audio signal metrics.
    
    Args:
        audio: Audio array
        
    Returns:
        Tuple of (peak_amplitude, rms_amplitude)
    """
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return peak, rms


def evaluate_signal(peak: float, rms: float) -> str:
    """Evaluate signal quality based on levels.
    
    Args:
        peak: Peak amplitude
        rms: RMS amplitude
        
    Returns:
        Assessment string
    """
    if peak > 0.05:
        return "✓ Signal looks GOOD!"
    elif peak > 0.01:
        return "⚠ Microphone works, but signal is somewhat quiet."
    else:
        return "✗ Signal is VERY quiet. Check microphone placement or levels."


def main() -> None:
    """Run microphone test."""
    args = parse_arguments()
    
    # Show device info
    try:
        logger.info("Querying device %d", args.device)
        device_info = sd.query_devices(args.device)
        print("Using microphone:")
        print(f"  Name: {device_info.get('name', 'Unknown')}")
        print(f"  Inputs: {device_info.get('max_input_channels', 0)}")
        print(f"  Sample rate: {device_info.get('default_samplerate', 0):.0f} Hz\n")
    except Exception as e:
        logger.warning("Could not get device info: %s", e)
    
    # Record audio
    audio = record_audio(args.device, args.duration, SAMPLE_RATE)
    
    # Analyze signal
    peak, rms = analyze_audio(audio)
    
    # Display results
    print("\n" + "=" * 60)
    print("Recording Analysis")
    print("=" * 60)
    print(f"Peak volume : {peak:.6f}")
    print(f"RMS volume  : {rms:.6f}\n")
    print(evaluate_signal(peak, rms))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        logger.info("Starting microphone test")
        main()
        logger.info("Microphone test completed successfully")
    except Exception as e:
        logger.error("Test failed: %s", e, exc_info=True)
        exit(1)