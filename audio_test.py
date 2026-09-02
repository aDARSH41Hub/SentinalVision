"""Test script for recording audio and testing AST audio classification model.

This script demonstrates basic audio recording with sounddevice and inference
using the AudioSet-tuned AST model. Useful for verifying model performance
and debugging audio-related issues.

Usage:
    python audio_test.py
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForAudioClassification, AutoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Configuration constants
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
DURATION = 5
MIC_DEVICE = 0
COMPUTE_DEVICE = "cpu"

# Gain to amplify captured waveform for better model input.
# NOTE: This is experimental and tuned for specific recording conditions.
# Not used in production pipeline.
GAIN = 5.0
TOP_K = 10


def load_model() -> Tuple[AutoProcessor, AutoModelForAudioClassification, torch.device]:
    """Load AST model and processor.
    
    Returns:
        Tuple of (processor, model, device)
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        logger.info("Loading AST processor...")
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        logger.info("Loading AST model...")
        model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        
        device = torch.device(COMPUTE_DEVICE)
        model.to(device)
        model.eval()
        
        logger.info("AST model loaded successfully")
        return processor, model, device
        
    except Exception as e:
        logger.error("Failed to load model: %s", e, exc_info=True)
        raise RuntimeError(f"Could not load AST model: {e}") from e


def record_audio(duration: int, sample_rate: int, device_id: int) -> np.ndarray:
    """Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        device_id: Microphone device index
        
    Returns:
        Audio array (mono, float32)
        
    Raises:
        RuntimeError: If recording fails
    """
    try:
        import sounddevice as sd
        
        logger.info("Recording for %d seconds...", duration)
        print("START! Speak continuously and clearly.")
        
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device_id,
        )
        
        sd.wait()
        audio = audio.squeeze()
        
        logger.info("Recording complete. Shape: %s", audio.shape)
        return audio
        
    except Exception as e:
        logger.error("Recording failed: %s", e, exc_info=True)
        raise RuntimeError(f"Could not record audio: {e}") from e


def analyze_audio(audio: np.ndarray) -> Tuple[float, float]:
    """Compute audio metrics.
    
    Args:
        audio: Audio array
        
    Returns:
        Tuple of (peak_amplitude, rms_amplitude)
    """
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return peak, rms


def run_inference(
    processor: AutoProcessor,
    model: AutoModelForAudioClassification,
    device: torch.device,
    audio: np.ndarray,
    sample_rate: int,
) -> List[Tuple[str, float]]:
    """Run AST inference on audio.
    
    Args:
        processor: Audio processor
        model: Loaded model
        device: Compute device
        audio: Audio samples
        sample_rate: Sample rate in Hz
        
    Returns:
        List of (label, probability) tuples for top-K classes
    """
    try:
        logger.info("Running AST inference...")
        
        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        top_values, top_indices = torch.topk(probabilities, k=TOP_K)
        
        results = []
        for value, index in zip(top_values, top_indices):
            class_id = int(index)
            label = model.config.id2label[class_id]
            prob = float(value)
            results.append((label, prob))
        
        logger.info("Inference complete")
        return results
        
    except Exception as e:
        logger.error("Inference failed: %s", e, exc_info=True)
        raise RuntimeError(f"Could not run inference: {e}") from e


def print_results(
    audio: np.ndarray,
    processed_audio: np.ndarray,
    predictions: List[Tuple[str, float]],
) -> None:
    """Print analysis and prediction results.
    
    Args:
        audio: Original audio
        processed_audio: Processed audio (after gain)
        predictions: Model predictions
    """
    raw_peak, raw_rms = analyze_audio(audio)
    processed_peak, processed_rms = analyze_audio(processed_audio)
    
    print("\n" + "=" * 60)
    print("AUDIO ANALYSIS")
    print("=" * 60)
    
    print("\nRAW AUDIO")
    print(f"  Peak: {raw_peak:.6f}")
    print(f"  RMS:  {raw_rms:.6f}")
    
    print("\nAFTER DIGITAL GAIN")
    print(f"  Gain: {GAIN}x")
    print(f"  Peak: {processed_peak:.6f}")
    print(f"  RMS:  {processed_rms:.6f}")
    
    print("\nTOP PREDICTIONS")
    print("-" * 60)
    for i, (label, prob) in enumerate(predictions, 1):
        print(f"  {i:2d}. {label:<40} {prob:.4f}")
    
    print("=" * 60 + "\n")


def main() -> None:
    """Main test routine."""
    logger.info("Starting audio test")
    
    # Load model
    processor, model, device = load_model()
    
    # Record audio
    audio = record_audio(DURATION, SAMPLE_RATE, MIC_DEVICE)
    
    # Apply gain and clip
    processed_audio = audio * GAIN
    processed_audio = np.clip(processed_audio, -1.0, 1.0)
    
    # Run inference
    predictions = run_inference(
        processor,
        model,
        device,
        processed_audio,
        SAMPLE_RATE,
    )
    
    # Display results
    print_results(audio, processed_audio, predictions)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Test failed: %s", e, exc_info=True)
        exit(1)