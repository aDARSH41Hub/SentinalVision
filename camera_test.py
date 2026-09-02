"""Basic webcam capture test.

Simple test script to verify that webcam is accessible and functioning.
Displays live video feed from the default camera. Press Q to exit.

Usage:
    python camera_test.py
"""

import logging
from typing import Optional

import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(camera_index: int = 0, window_title: str = "Webcam Test") -> None:
    """Run webcam capture test.
    
    Args:
        camera_index: Camera device index (default: 0)
        window_title: Window title for display
        
    Raises:
        RuntimeError: If camera cannot be opened
    """
    logger.info("Opening camera device %d", camera_index)
    
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        logger.error("Failed to open camera device %d", camera_index)
        raise RuntimeError(f"Could not open webcam at index {camera_index}")
    
    logger.info("Camera opened successfully")
    logger.info("Press Q to exit")
    
    frame_count = 0
    frame_errors = 0
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret:
                frame_errors += 1
                logger.warning("Failed to read frame (error %d)", frame_errors)
                if frame_errors > 10:
                    logger.error("Too many consecutive frame read failures")
                    break
                continue
            
            frame_errors = 0
            frame_count += 1
            
            # Display frame
            cv2.imshow(window_title, frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested by user")
                break
    
    finally:
        logger.info("Closing camera")
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Captured %d frames successfully", frame_count)


if __name__ == "__main__":
    try:
        logger.info("Starting camera test")
        main()
        logger.info("Camera test completed successfully")
    except Exception as e:
        logger.error("Camera test failed: %s", e, exc_info=True)
        exit(1)