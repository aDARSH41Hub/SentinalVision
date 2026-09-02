# SentinelVision Quick Start Guide

After code quality enhancements - Getting started with the improved codebase.

## 🚀 Quick Start

### Run Audio Detector (Live)
```bash
cd d:\SentinalVision
python audio_live_test.py
# Output: Real-time threat detection results with logging
```

### Run Vision Pipeline
```bash
python vision.py
# Output: Live webcam with YOLO + Haar detections and FPS
```

### Test Individual Components
```bash
# Audio: Test model with recorded audio
python audio_test.py

# Vision: Test YOLO on sample image
python yolo_test.py

# Camera: Test webcam connectivity
python camera_test.py

# Face Detection: Test Haar cascades
python face_test.py

# YOLO Webcam: Real-time object detection
python webcam_yolo.py

# Microphone: Test microphone signal levels
python mic_test.py --device 0 --duration 5

# List devices: See available microphones
python mic_devices.py
```

---

## 📋 Key Improvements You'll Notice

### 1. **Type Safety**
All functions now have complete type hints. Your IDE will help you:
- ✅ Autocomplete function parameters
- ✅ Catch type errors before runtime
- ✅ See expected return types

### 2. **Better Documentation**
Every function has clear docstrings:
```python
def process(frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Process frame through detectors.
    
    Args:
        frame: Input frame (BGR format)
        
    Returns:
        (annotated_frame, detection_results)
        
    Raises:
        RuntimeError: If inference fails
    """
```

### 3. **Structured Logging**
Replace confusing print output with clear, structured logs:
```
2026-09-01 10:23:45,123 - audio - INFO - Loading AST processor from MIT/ast-finetuned-audioset-10-10-0.4593
2026-09-01 10:23:46,456 - audio - INFO - AST audio detector loaded successfully
2026-09-01 10:23:50,789 - audio - WARNING - Threat detected: Gunshot, gunfire (score=0.850, risk=THREATENING)
```

### 4. **Robust Error Handling**
Clear error messages instead of cryptic exceptions:
```
ERROR - Failed to load YOLO model: FileNotFoundError: [Errno 2] No such file or directory: 'yolo26n.pt'
ERROR - Could not open webcam at index 0: RuntimeError: [-215:Assertion failed] !_src.empty() in function 'cvtColor'
```

### 5. **Configuration Classes**
Easy to customize detector parameters:
```python
from audio import AudioDetectorConfig, AudioDetector

config = AudioDetectorConfig()
config.SAMPLE_RATE = 8000  # Custom sampling rate
config.THREAT_SCORE_SUSPICIOUS = 0.25  # Lower threshold
detector = AudioDetector(config=config)
```

---

## 📚 Understanding the Code

### Core Modules

#### `audio.py` - Audio Threat Detection
```python
from audio import AudioDetector

# Create detector
detector = AudioDetector()

# Get latest result in a loop
while True:
    result = detector.get_result()
    print(f"Label: {result['label']}")
    print(f"Risk: {result['risk_label']}")  # BENIGN, SUSPICIOUS, THREATENING
    print(f"Threat Score: {result['threat_score']:.3f}")
    time.sleep(1)

detector.stop()
```

**Key Features:**
- Runs inference in background thread (non-blocking)
- Real-time circular audio buffering
- Threat scoring with weighted classification
- 24 threat classes (gunshots, explosions, screaming, alarms, etc.)

#### `vision.py` - Vision Detection Pipeline
```python
from vision import VisionDetector
import cv2

# Create detector
detector = VisionDetector(
    yolo_device="auto",      # GPU if available
    yolo_imgsz=320,           # Fast but accurate
    haar_scale_width=480      # Downscale for speed
)

# Process frames
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    annotated, stats = detector.process(frame)
    
    print(f"Persons: {stats['person_count']}")
    print(f"Faces: {stats['face_count']}")
    print(f"FPS: {stats['fps']:.1f}")
    
    cv2.imshow("Vision", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
```

**Key Features:**
- Dual detector: YOLO (general objects) + Haar (faces)
- Automatic GPU/CPU selection
- Intelligent downscaling (4-9x Haar speedup)
- Real-time FPS monitoring

### Test/Utility Modules

All test files now have:
- ✅ Type hints
- ✅ Docstrings
- ✅ Structured logging
- ✅ Error recovery
- ✅ `__main__` blocks for standalone testing
- ✅ CLI arguments (where applicable)

---

## 🔍 Understanding Types

### Common Type Patterns in SentinelVision

```python
# Audio results
result: Dict[str, Any] = {
    "label": str,                              # Top prediction
    "confidence": float,                       # 0-1
    "threat_score": float,                     # 0-1
    "risk_label": str,                         # BENIGN/SUSPICIOUS/THREATENING
    "peak": float,                             # Peak amplitude
    "rms": float,                              # RMS amplitude
    "top_predictions": List[Tuple[str, float]],
    "threat_predictions": List[Tuple[str, float, float, float]],
}

# Vision results
features: Dict[str, Any] = {
    "person_count": int,
    "face_count": int,
    "objects": List[Tuple[str, float]],        # (class_name, confidence)
    "fps": float,
}

# Optional parameters
config: Optional[AudioDetectorConfig] = None
device: Optional[str] = "cuda"
```

---

## 🛠️ Customization Examples

### Custom Audio Config
```python
from audio import AudioDetector, AudioDetectorConfig

class MyConfig(AudioDetectorConfig):
    SAMPLE_RATE = 8000          # Lower sample rate for speed
    BUFFER_SECONDS = 1          # Shorter buffer
    THREAT_SCORE_SUSPICIOUS = 0.25  # More sensitive

detector = AudioDetector(config=MyConfig())
```

### Custom Vision Config
```python
from vision import VisionDetector, VisionDetectorConfig

config = VisionDetectorConfig()
config.YOLO_IMGSZ = 640         # Higher accuracy, slower
config.HAAR_SCALE_WIDTH = 320   # More downscaling = faster
config.YOLO_CONF_THRESHOLD = 0.5  # Stricter confidence

detector = VisionDetector(config=config)
```

### Changing Log Levels
```python
import logging

# Show all debug messages
logging.basicConfig(level=logging.DEBUG)

# Show only warnings and errors
logging.basicConfig(level=logging.WARNING)

detector = AudioDetector()  # Now logs at configured level
```

---

## 🐛 Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now you'll see:
# - Model loading details
# - Frame processing timing
# - Buffer state information
# - Device selection choices
```

### Check Device Availability
```python
python mic_devices.py   # List all audio devices
python camera_test.py   # Test camera at index 0
```

### Verify Models Exist
```bash
# Make sure yolo26n.pt is in d:\SentinalVision\
ls -la d:\SentinalVision\yolo26n.pt
```

### Test Microphone Signal
```bash
python mic_test.py --device 0
# Shows peak amplitude and quality assessment
```

---

## 📊 Integration Example

Combine audio and vision detection:

```python
import logging
import time
from audio import AudioDetector
from vision import VisionDetector
import cv2

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

audio = AudioDetector()
vision = VisionDetector(yolo_device="auto")
camera = cv2.VideoCapture(0)

logger.info("Multi-modal detection started")

try:
    while True:
        # Get audio result (runs in background)
        audio_result = audio.get_result()
        
        # Get vision result
        ret, frame = camera.read()
        if ret:
            vis_frame, vis_stats = vision.process(frame)
            cv2.imshow("Detection", vis_frame)
        
        # Combined analysis
        if audio_result['risk_label'] == 'THREATENING':
            logger.warning("AUDIO THREAT: %s", audio_result['label'])
            
        if vis_stats['person_count'] > 3:
            logger.info("CROWD: %d persons detected", 
                       vis_stats['person_count'])
        
        time.sleep(0.1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    audio.stop()
    camera.release()
    cv2.destroyAllWindows()
    logger.info("Detection stopped")
```

---

## 📋 Checklist: Before Using in Production

- [ ] Run `python audio_live_test.py` - Verify audio detection works
- [ ] Run `python vision.py` - Verify vision detection works
- [ ] Check `python mic_devices.py` - Verify microphone device index
- [ ] Test `python mic_test.py --device X` - Verify signal levels
- [ ] Review `ENHANCEMENTS.md` - Understand what changed
- [ ] Review `CODE_STANDARDS.md` - Understand new patterns
- [ ] Check logs for errors - Look for ResourceWarning, etc.
- [ ] Monitor FPS - Ensure real-time performance (30+ FPS)
- [ ] Test error scenarios - Unplug camera, disconnect mic, etc.

---

## 🔧 Troubleshooting

### Issue: "Could not load YOLO model: FileNotFoundError"
**Solution**: Ensure `yolo26n.pt` is in `d:\SentinalVision\`

### Issue: "Could not open webcam at index 0"
**Solution**: 
```bash
python camera_test.py
# Try different indices if needed
```

### Issue: "Could not open microphone device 0"
**Solution**:
```bash
python mic_devices.py
# Find correct device index, then use:
python mic_test.py --device N
```

### Issue: Very low FPS or laggy detection
**Solution**:
- Reduce YOLO inference size: `yolo_imgsz=160`
- Increase Haar downscaling: `haar_scale_width=320`
- Use GPU if available: `yolo_device="auto"`

### Issue: Missing threat detections
**Solution**:
- Lower threat thresholds in Config
- Check microphone signal: `python mic_test.py`
- Review `top_predictions` in audio results

---

## 📖 Documentation Files

- **[ENHANCEMENTS.md](ENHANCEMENTS.md)** - Detailed before/after for each file
- **[CODE_STANDARDS.md](CODE_STANDARDS.md)** - Coding patterns and best practices
- **[audio.py](audio.py)** - See module docstring for audio architecture
- **[vision.py](vision.py)** - See module docstring for vision architecture

---

## 🎯 Next Steps

### For Understanding the Code
1. Read module docstrings (top of each .py file)
2. Review class docstrings (class definitions)
3. Check function docstrings (parameter/return details)

### For Making Changes
1. Follow type hint patterns from [CODE_STANDARDS.md](CODE_STANDARDS.md)
2. Use logging instead of print()
3. Add docstrings to new functions
4. Extract constants to Config classes
5. Ensure backward compatibility

### For Contributing
1. Run tests: `python module_name.py` (uses __main__)
2. Check logs for warnings: look for "WARNING" or "ERROR"
3. Verify type hints with IDE (if configured with mypy)
4. Review [ENHANCEMENTS.md](ENHANCEMENTS.md) to match patterns

---

## 📞 Key Contacts/Resources

- **Python Docs**: https://docs.python.org/3/library/typing.html
- **OpenCV Docs**: https://docs.opencv.org/
- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **HuggingFace Transformers**: https://huggingface.co/transformers/

---

## Version Info

- **Enhancement Date**: 2026-09-01
- **Python Version**: 3.8+ (required for type hints)
- **Compatibility**: Fully backward compatible (no breaking changes)

All functionality preserved. Only improvements to code quality, documentation, and error handling.

Happy coding! 🚀
