import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import argparse
import os

# Constants
MODEL_PATH = 'C:/Users/ashis/OneDrive/Desktop/MID/morph_detection_model.keras'
IMG_SIZE = (224, 224)
THRESHOLD = 0.50  # Confidence threshold (adjust as needed)

# Load model globally for reuse
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def load_and_preprocess(image_path):
    """Load and preprocess an image for prediction"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize to [0,1]
    return np.expand_dims(img, axis=0)

def predict_image(image_path, threshold=THRESHOLD):
    """Make prediction on a single image"""
    try:
        if model is None:
            raise ValueError("Model not loaded.")
        processed_img = load_and_preprocess(image_path)
        confidence = model.predict(processed_img)[0][0]
        status = "MORPHED" if confidence > threshold else "ORIGINAL"
        actual_confidence = confidence if status == "MORPHED" else 1 - confidence
        return {
            "status": status,
            "confidence": float(actual_confidence),
            "raw_output": float(confidence)
        }
    except Exception as e:
        return {"error": str(e)}

def evaluate_test_set(test_dir):
    """Evaluate all images in a test directory"""
    results = []
    for label in ['morphed', 'original']:
        label_dir = os.path.join(test_dir, label)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            result = predict_image(img_path)
            result['filename'] = img_name
            result['true_label'] = label.upper()
            results.append(result)
    return results

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to single image for prediction')
    parser.add_argument('--test-dir', help='Path to directory with test images')
    args = parser.parse_args()

    if args.image:
        result = predict_image(args.image)
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("\nPrediction Result:")
            print(f"Status: {result['status']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Raw Output: {result['raw_output']:.4f}")

    elif args.test_dir:
        results = evaluate_test_set(args.test_dir)
        correct = sum(1 for r in results if r['status'] == r['true_label'])
        accuracy = correct / len(results)
        print(f"\nTest Set Evaluation ({len(results)} images):")
        print(f"Accuracy: {accuracy:.2%}")
        print("\nSample predictions:")
        for r in results[:5]:
            print(f"{r['filename']}: Pred={r['status']}, True={r['true_label']}, Conf={r['confidence']:.2%}")
    else:
        print("Please specify either --image or --test-dir")
        print("Example:")
        print("  python detect.py --image datasets/test/original/test1.jpg")
        print("  python detect.py --test-dir datasets/test")

# CLI entrypoint
if __name__ == "__main__":
    cli_main()
