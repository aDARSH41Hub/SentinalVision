# SentinelVision Code Standards & Patterns

Quick reference guide for understanding and maintaining the enhanced SentinelVision codebase.

## Type Hints

All functions must have complete type hints following Python 3.8+ standards.

### Example Pattern

```python
from typing import Optional, List, Dict, Tuple, Any

def process_frame(
    frame: np.ndarray,
    model: YOLO,
    confidence: float = 0.45,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Process video frame through detector.
    
    Args:
        frame: Input frame in BGR format
        model: Loaded YOLO model instance
        confidence: Detection confidence threshold [0-1]
        
    Returns:
        Tuple of:
            - Annotated frame with boxes
            - Dictionary with detection results
            
    Raises:
        ValueError: If confidence not in [0, 1]
        RuntimeError: If inference fails
    """
    if not 0 <= confidence <= 1:
        raise ValueError(f"Confidence must be [0-1], got {confidence}")
    
    try:
        results = model(frame, conf=confidence)
        return results[0].plot(), {
            "boxes": len(results[0].boxes),
            "fps": 30.0,
        }
    except Exception as e:
        logger.error("Inference failed: %s", e, exc_info=True)
        raise RuntimeError(f"Could not process frame: {e}") from e
```

### Type Hints Quick Reference

```python
# Optional parameters
param: Optional[int] = None

# Lists, dicts, tuples
param: List[str]
param: Dict[str, float]
param: Tuple[int, str, bool]

# Union types (pick one)
param: Union[str, int]  # or (Python 3.10+): param: str | int

# Callables
param: Callable[[int, str], bool]

# Returns
def func() -> None:           # No return
def func() -> str:            # Single value
def func() -> Tuple[int, str]: # Multiple values
def func() -> Optional[Dict]:  # Can be None
```

---

## Documentation (Docstrings)

Use Google-style docstrings for all public functions and classes.

### Module-Level Docstring

```python
"""Real-time video processing module for threat detection.

This module provides efficient multi-detector vision processing for security
surveillance using YOLO and Haar cascades.

Key Features:
    - Real-time YOLO object detection (GPU-accelerated)
    - Haar cascade face detection with downscaling
    - FPS monitoring and performance diagnostics
    - Configurable detector parameters
    - Robust error handling and recovery

Performance Notes:
    - Haar detection runs on downscaled frames (4-9x speedup)
    - YOLO inference size tuned to 320x320 for real-time
    
Example:
    >>> detector = VisionDetector(yolo_device="auto")
    >>> camera = cv2.VideoCapture(0)
    >>> ret, frame = camera.read()
    >>> annotated, stats = detector.process(frame)
    >>> print(f"Detected {stats['person_count']} persons")
"""
```

### Function Docstring

```python
def detect_faces(
    frame: np.ndarray,
    scale: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """Detect faces in image using Haar cascades.
    
    Runs Haar cascade detector on downscaled frame for efficiency,
    then maps results back to original resolution.
    
    Args:
        frame: Input frame (BGR numpy array)
        scale: Downscaling factor [0-1] (smaller = faster)
        
    Returns:
        List of face rectangles as (x, y, width, height) tuples
        
    Raises:
        ValueError: If scale not in (0, 1]
        RuntimeError: If cascade fails to load
        
    Example:
        >>> frame = cv2.imread("image.jpg")
        >>> faces = detect_faces(frame, scale=0.5)
        >>> print(f"Found {len(faces)} faces")
    """
```

### Class Docstring

```python
class VisionDetector:
    """Multi-detector vision processor (YOLO + Haar cascades).
    
    Efficiently processes video frames through both YOLO and Haar cascade
    detectors with real-time FPS monitoring and configurable parameters.
    
    Attributes:
        yolo: Loaded YOLO model instance
        face_cascade: Loaded Haar cascade classifier
        device: PyTorch device ('cpu' or 'cuda')
        fps: Current frames-per-second estimate
        
    Example:
        >>> detector = VisionDetector(yolo_device="auto")
        >>> annotated, stats = detector.process(frame)
        >>> print(f"FPS: {stats['fps']:.1f}")
    """
```

---

## Error Handling

Follow these patterns for robust error handling.

### Try-Except-Finally Pattern

```python
def process_video(video_path: str) -> None:
    """Process video file."""
    camera = None
    
    try:
        logger.info("Opening video: %s", video_path)
        camera = cv2.VideoCapture(video_path)
        
        if not camera.isOpened():
            raise FileNotFoundError(f"Could not open: {video_path}")
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            logger.debug("Processing frame")
            # ... process frame
    
    except FileNotFoundError as e:
        logger.error("Video file not found: %s", e)
        raise  # Re-raise for caller to handle
    
    except Exception as e:
        logger.error("Unexpected error during processing: %s", e, exc_info=True)
        raise RuntimeError(f"Video processing failed: {e}") from e
    
    finally:
        if camera is not None:
            logger.info("Closing video stream")
            camera.release()
```

### Specific Exception Types

```python
# For user input validation
if not 0 <= confidence <= 1:
    raise ValueError(f"Confidence must be [0-1], got {confidence}")

# For missing resources
if not Path(model_path).exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

# For initialization failures
if model is None:
    raise RuntimeError("Could not load model")

# For external library errors
try:
    result = dangerous_operation()
except (RuntimeError, OSError) as e:
    logger.error("Operation failed: %s", e, exc_info=True)
    raise RuntimeError("User-friendly error message") from e
```

---

## Logging

Replace all `print()` statements with structured logging.

### Setup in Main

```python
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("Starting detector")
    try:
        detector = AudioDetector()
        # ... use detector
    except Exception as e:
        logger.error("Failed to initialize: %s", e, exc_info=True)
        exit(1)
```

### Logging Levels

```python
logger.debug("Detailed diagnostic info")              # Development only
logger.info("Important state changes")                 # Normal operation
logger.warning("Recoverable issues, important alerts") # Needs attention
logger.error("Serious errors, needs investigation")   # Failure states
logger.critical("System is broken")                    # Rare

# Example usage
logger.info("Starting inference with batch_size=%d", batch_size)
logger.warning("GPU not available, falling back to CPU")
logger.error("Failed to load model from %s: %s", path, e, exc_info=True)
logger.debug("Inference completed in %.2f ms", elapsed_ms)
```

### Exception Logging

```python
# Always include exc_info=True for exception context
try:
    result = dangerous_call()
except Exception as e:
    logger.error("Operation failed: %s", e, exc_info=True)  # Includes traceback
    raise RuntimeError("User message") from e
```

---

## Configuration Management

Extract magic numbers into Config classes.

### Config Class Pattern

```python
class AudioDetectorConfig:
    """Configuration for audio detector.
    
    All constants are documented with their purpose and units.
    Can be subclassed to override defaults.
    """
    
    # Model configuration
    MODEL_NAME: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    
    # Audio parameters (Hz, seconds)
    SAMPLE_RATE: int = 16000
    BUFFER_SECONDS: int = 2
    BLOCKSIZE: int = 1600
    INFERENCE_INTERVAL_SECONDS: float = 1.0
    
    # Detection thresholds
    SILENCE_THRESHOLD: float = 0.01
    THREAT_SCORE_THREATENING: float = 0.70
    THREAT_SCORE_SUSPICIOUS: float = 0.30
    
    # Classification weights (0-1 scale)
    THREAT_CLASSES: Dict[str, float] = {
        "Gunshot, gunfire": 1.00,
        "Explosion": 1.00,
        # ... more classes
    }
```

### Usage in Class

```python
class AudioDetector:
    def __init__(
        self,
        config: Optional[AudioDetectorConfig] = None,
    ) -> None:
        """Initialize with optional custom config."""
        self.config = config or AudioDetectorConfig()
        
        # Use config values
        self.buffer_size = int(
            self.config.SAMPLE_RATE * self.config.BUFFER_SECONDS
        )
        
        if peak < self.config.SILENCE_THRESHOLD:
            logger.debug("Silence detected")
```

---

## Code Organization

### File Structure

```python
"""Module docstring with overview and examples."""

import logging
from pathlib import Path
from typing import ...

import numpy  # third-party
import cv2

logger = logging.getLogger(__name__)


class ConfigClass:
    """Configuration constants."""
    CONSTANT: type = value


class MainClass:
    """Primary class."""
    
    def __init__(self) -> None:
        """Initialize."""
    
    def public_method(self) -> None:
        """Public method."""
    
    def _private_method(self) -> None:
        """Private method."""


def helper_function() -> None:
    """Module-level helper."""


def main() -> None:
    """Entry point."""


if __name__ == "__main__":
    logging.basicConfig(...)
    try:
        main()
    except Exception as e:
        logger.error("Failed: %s", e, exc_info=True)
        exit(1)
```

### Import Organization

```python
# 1. Standard library (sorted alphabetically)
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

# 2. Third-party (sorted alphabetically)
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 3. Local imports
from audio import AudioDetector
from config import DetectorConfig

logger = logging.getLogger(__name__)
```

---

## Common Patterns

### Resource Cleanup

```python
def process_video(path: str) -> None:
    """Process video with guaranteed cleanup."""
    camera = None
    
    try:
        camera = cv2.VideoCapture(path)
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open: {path}")
        # ... process
    finally:
        if camera is not None:
            camera.release()
```

### Context Manager Alternative

```python
from contextlib import contextmanager

@contextmanager
def open_camera(index: int):
    """Context manager for camera access."""
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    try:
        yield camera
    finally:
        camera.release()

# Usage
with open_camera(0) as camera:
    ret, frame = camera.read()
```

### Device Selection

```python
def get_device(device_hint: str = "auto") -> str:
    """Detect GPU availability with fallback.
    
    Args:
        device_hint: 'auto', 'cpu', or 'cuda'
        
    Returns:
        Device string ('cpu' or 'cuda')
    """
    if device_hint != "auto":
        return device_hint
    
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        logger.warning("PyTorch not available, using CPU")
        return "cpu"
```

### FPS Calculation

```python
class FrameRateCounter:
    """Rolling 1-second FPS counter."""
    
    def __init__(self) -> None:
        self.frame_count = 0
        self.last_update = time.time()
        self.fps = 0.0
    
    def update(self) -> None:
        """Call once per frame."""
        self.frame_count += 1
        now = time.time()
        
        if now - self.last_update >= 1.0:
            self.fps = self.frame_count / (now - self.last_update)
            self.frame_count = 0
            self.last_update = now
```

---

## Testing Your Code

### Running Individual Modules

```bash
# Each module has a __main__ block for testing
python audio.py
python vision.py
python camera_test.py

# Some accept command-line arguments
python mic_test.py --device 1 --duration 3
python haar_face_test.py --camera 0 --blur
```

### Checking Logs

```python
# Set log level for debugging
logging.basicConfig(level=logging.DEBUG)  # See all messages

# Check production logs
logging.basicConfig(level=logging.INFO)   # Normal operation
```

---

## Checklist for New Code

When adding new features to SentinelVision, ensure:

- [ ] Function has complete type hints (parameters + return type)
- [ ] Function has docstring with Args/Returns/Raises sections
- [ ] Configuration constants extracted to Config class
- [ ] All print() replaced with logger.info/warning/error
- [ ] Specific exception types (ValueError, RuntimeError, etc.)
- [ ] Resource cleanup in finally blocks
- [ ] Logging at INFO level for important events
- [ ] Logging at WARNING level for recoverable errors
- [ ] Unit tests or __main__ test code included
- [ ] __all__ exports documented (if applicable)

---

## Questions?

Refer to specific files for working examples:
- Type hints: [audio.py](audio.py), [vision.py](vision.py)
- Documentation: [audio_live_test.py](audio_live_test.py), [vision.py](vision.py)
- Error handling: [audio_test.py](audio_test.py), [camera_test.py](camera_test.py)
- Logging: All files (replace print with logger)
- Configuration: [audio.py](audio.py), [vision.py](vision.py)

See [ENHANCEMENTS.md](ENHANCEMENTS.md) for detailed before/after examples.
