"""Real-time YOLO object detection on webcam feed.

Continuously captures frames from webcam and runs optimized YOLO inference.
Displays results with FPS counter and device information. Press Q to exit.

Optimizations:
    1. Device selection: Automatically uses GPU if available, fallback to CPU
    2. Inference size tuning: 320x320 (4x faster than 640x640 default)
    3. Camera parameter optimization: Buffer size 1, FPS 30
    4. Error recovery: Automatic frame read retry with timeout

Usage:
    python webcam_yolo.py
"""

import logging
import time
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "yolo26n.pt"
YOLO_DEVICE = "auto"
YOLO_IMGSZ = 320
YOLO_CONF_THRESHOLD = 0.45

CAMERA_INDEX = 0
CAMERA_BUFFER_SIZE = 1
CAMERA_FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

FRAME_RECOVERY_MAX_FAILURES = 10


def get_device() -> str:
    """Auto-detect GPU availability, fallback to CPU.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    try:
        import torch
        if YOLO_DEVICE == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = YOLO_DEVICE
        return device
    except ImportError:
        logger.warning("PyTorch not available, using CPU")
        return "cpu"


def load_model(model_path: str) -> YOLO:
    """Load YOLO model.
    
    Args:
        model_path: Path to YOLO model file
        
    Returns:
        Loaded YOLO model
        
    Raises:
        RuntimeError: If model cannot be loaded
    """
    try:
        logger.info("Loading YOLO model: %s", model_path)
        model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", e, exc_info=True)
        raise RuntimeError(f"Could not load YOLO model: {e}") from e


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open and configure camera for optimal performance.
    
    Args:
        camera_index: Camera device index
        
    Returns:
        Opened VideoCapture object
        
    Raises:
        RuntimeError: If camera cannot be opened
    """
    try:
        logger.info("Opening camera device %d", camera_index)
        camera = cv2.VideoCapture(camera_index)
        
        if not camera.isOpened():
            raise RuntimeError("Camera failed to open")
        
        # Optimize for real-time capture
        camera.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        logger.info("Camera configured: %dx%d, %d FPS", CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        return camera
        
    except Exception as e:
        logger.error("Failed to open camera: %s", e, exc_info=True)
        raise RuntimeError(f"Could not open webcam at index {camera_index}: {e}") from e


def main() -> None:
    """Run YOLO webcam detection."""
    # Load model
    device = get_device()
    model = load_model(MODEL_PATH)
    
    logger.info("YOLO device: %s", device)
    logger.info("YOLO inference size: %dx%d", YOLO_IMGSZ, YOLO_IMGSZ)
    
    # Open camera
    camera = open_camera(CAMERA_INDEX)
    
    logger.info("Starting YOLO webcam processing")
    logger.info("Press Q to quit")
    
    # FPS tracking
    frame_count = 0
    start_time = time.time()
    last_fps_update = time.time()
    fps = 0.0
    frame_read_failures = 0
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret:
                frame_read_failures += 1
                if frame_read_failures > FRAME_RECOVERY_MAX_FAILURES:
                    logger.error(
                        "Could not read frame %d times. Exiting.",
                        FRAME_RECOVERY_MAX_FAILURES,
                    )
                    break
                logger.warning("Frame read failed (attempt %d)", frame_read_failures)
                continue
            
            frame_read_failures = 0
            
            # Run YOLO inference
            try:
                results = model(
                    frame,
                    verbose=False,
                    device=device,
                    imgsz=YOLO_IMGSZ,
                    conf=YOLO_CONF_THRESHOLD,
                )
                annotated = results[0].plot()
            except Exception as e:
                logger.warning("YOLO inference failed: %s", e)
                annotated = frame.copy()
            
            # Update FPS counter
            frame_count += 1
            now = time.time()
            if now - last_fps_update >= 1.0:
                fps = frame_count / (now - last_fps_update)
                frame_count = 0
                last_fps_update = now
            
            # Draw performance info
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"Device: {device.upper()}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            
            # Display frame
            cv2.imshow("YOLO Webcam (Optimized)", annotated)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested by user")
                break
    
    finally:
        logger.info("Cleaning up resources")
        camera.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        logger.info("YOLO test finished. Average FPS: %.1f", avg_fps)


if __name__ == "__main__":
    try:
        logger.info("Starting YOLO webcam test")
        main()
        logger.info("Webcam test completed successfully")
    except Exception as e:
        logger.error("Test failed: %s", e, exc_info=True)
        exit(1)