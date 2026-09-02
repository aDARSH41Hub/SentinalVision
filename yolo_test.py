"""Basic YOLO object detection test.

Simple test script that loads YOLOv8 model and runs inference on
a sample image from the internet. Demonstrates object detection
and result display.

Usage:
    python yolo_test.py
"""

import logging
from typing import List, Tuple

from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "yolo26n.pt"
TEST_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


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


def run_inference(model: YOLO, image_source: str) -> List[Tuple[str, float]]:
    """Run YOLO inference on image.
    
    Args:
        model: Loaded YOLO model
        image_source: Image file path or URL
        
    Returns:
        List of (class_name, confidence) tuples
        
    Raises:
        RuntimeError: If inference fails
    """
    try:
        logger.info("Running inference on: %s", image_source)
        results = model(image_source)
        logger.info("Inference complete")
        
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                detections.append((class_name, confidence))
        
        return detections
        
    except Exception as e:
        logger.error("Inference failed: %s", e, exc_info=True)
        raise RuntimeError(f"Could not run inference: {e}") from e


def print_results(detections: List[Tuple[str, float]]) -> None:
    """Print detection results.
    
    Args:
        detections: List of detections from run_inference
    """
    print("\n" + "=" * 60)
    print("YOLO Detection Results")
    print("=" * 60)
    
    if not detections:
        print("No objects detected.")
    else:
        print(f"\nDetected {len(detections)} object(s):\n")
        for class_name, confidence in detections:
            print(f"  • {class_name:<20} ({confidence:.2%})")
    
    print("=" * 60 + "\n")


def main() -> None:
    """Run YOLO detection test."""
    # Load model
    model = load_model(MODEL_PATH)
    
    # Run inference
    detections = run_inference(model, TEST_IMAGE_URL)
    
    # Display results
    print_results(detections)


if __name__ == "__main__":
    try:
        logger.info("Starting YOLO test")
        main()
        logger.info("YOLO test completed successfully")
    except Exception as e:
        logger.error("YOLO test failed: %s", e, exc_info=True)
        exit(1)