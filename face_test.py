"""Basic Haar cascade face detection test.

Simple test script to verify Haar cascade face detection works correctly.
Displays live video feed with detected faces boxed.

Usage:
    python face_test.py
"""

import logging

import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
HAAR_SCALE_FACTOR = 1.05
HAAR_MIN_NEIGHBORS = 3
HAAR_MIN_SIZE = 40
CAMERA_INDEX = 0


def load_haar_cascade() -> cv2.CascadeClassifier:
    """Load Haar cascade face detector.
    
    Returns:
        Loaded cascade classifier
        
    Raises:
        RuntimeError: If cascade cannot be loaded
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    
    if cascade.empty():
        logger.error("Failed to load Haar cascade from %s", cascade_path)
        raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
    
    logger.info("Haar cascade loaded successfully")
    return cascade


def main() -> None:
    """Run Haar face detection test.
    
    Raises:
        RuntimeError: If cascade or camera cannot be initialized
    """
    logger.info("Initializing Haar face detector")
    face_cascade = load_haar_cascade()
    
    logger.info("Opening camera device %d", CAMERA_INDEX)
    camera = cv2.VideoCapture(CAMERA_INDEX)
    
    if not camera.isOpened():
        logger.error("Failed to open camera device %d", CAMERA_INDEX)
        raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}")
    
    logger.info("Camera opened successfully")
    logger.info("Press Q to exit")
    
    frame_count = 0
    frame_errors = 0
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret:
                frame_errors += 1
                logger.warning("Failed to read frame (attempt %d)", frame_errors)
                if frame_errors > 10:
                    logger.error("Too many consecutive frame read failures")
                    break
                continue
            
            frame_errors = 0
            frame_count += 1
            
            # Convert to grayscale and equalize histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Detect faces
            try:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=HAAR_SCALE_FACTOR,
                    minNeighbors=HAAR_MIN_NEIGHBORS,
                    minSize=(HAAR_MIN_SIZE, HAAR_MIN_SIZE),
                )
            except Exception as e:
                logger.warning("Face detection failed: %s", e)
                faces = []
            
            # Draw detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    "Face",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
            
            # Display face count
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            
            # Show result
            cv2.imshow("Haar Face Detection", frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested by user")
                break
    
    finally:
        logger.info("Cleaning up resources")
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Test completed. Processed %d frames", frame_count)


if __name__ == "__main__":
    try:
        logger.info("Starting Haar face detection test")
        main()
        logger.info("Test finished successfully")
    except Exception as e:
        logger.error("Test failed: %s", e, exc_info=True)
        exit(1)