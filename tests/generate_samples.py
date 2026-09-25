import cv2
import numpy as np
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import WireframePreprocessor

def create_synthetic_wireframe(filename: str = "sample_wireframe.png") -> str:
    """Generates a synthetic hand-drawn style UI wireframe image with uneven lighting and low contrast."""
    h, w = 600, 800
    img = np.full((h, w), 240, dtype=np.uint8)  # Light gray paper background

    # Add artificial uneven lighting / vignette noise
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    vignette = 255 - np.uint8(30 * (X**2 + Y**2))
    img = cv2.multiply(img, vignette // 255)

    # 1. Header Navigation Bar Wireframe
    cv2.rectangle(img, (40, 30), (760, 90), (60, 60, 60), 2)
    cv2.putText(img, "[ Logo ]", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    cv2.rectangle(img, (550, 45), (630, 75), (70, 70, 70), 2)  # Nav link 1
    cv2.rectangle(img, (650, 45), (740, 75), (40, 40, 40), 2)  # Nav link 2

    # 2. Main Banner Box
    cv2.rectangle(img, (40, 120), (760, 260), (50, 50, 50), 2)
    cv2.putText(img, "Hero Header Sketch", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    cv2.rectangle(img, (70, 200), (220, 240), (30, 30, 30), 2) # Call to Action Button

    # 3. Card Container 1
    cv2.rectangle(img, (40, 290), (380, 540), (50, 50, 50), 2)
    cv2.rectangle(img, (60, 310), (360, 420), (80, 80, 80), 2) # Card Image Box
    cv2.rectangle(img, (60, 440), (360, 470), (60, 60, 60), 2) # Card Title Box
    cv2.rectangle(img, (60, 485), (200, 520), (30, 30, 30), 2) # Card Action Button

    # 4. Card Container 2
    cv2.rectangle(img, (420, 290), (760, 540), (50, 50, 50), 2)
    cv2.rectangle(img, (440, 310), (740, 420), (80, 80, 80), 2) # Card Image Box
    cv2.rectangle(img, (440, 440), (740, 470), (60, 60, 60), 2) # Card Title Box
    cv2.rectangle(img, (440, 485), (580, 520), (30, 30, 30), 2) # Card Action Button

    # Add Gaussian noise to simulate camera capture / sketch texture
    noise = np.random.normal(0, 8, (h, w)).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    save_path = os.path.join(os.path.dirname(__file__), filename)
    cv2.imwrite(save_path, img_noisy)
    print(f"[OK] Generated synthetic wireframe at: {save_path}")
    return save_path

if __name__ == "__main__":
    wireframe_path = create_synthetic_wireframe("sample_wireframe.png")

    output_dir = os.path.join(os.path.dirname(__file__), "sample_outputs")
    processor = WireframePreprocessor(target_size=(800, 800))
    result = processor.process(wireframe_path, output_dir=output_dir)

    print("\n=== Pre-processing Execution Summary ===")
    print(f"Original Image Size: {result['original_shape']}")
    print(f"Target Normalized Size: {result['target_shape']}")
    print(f"Total UI Components Detected: {result['total_components']}")
    print(f"Processed images saved to directory: {output_dir}")
    print("\nSample Bounding Box Coordinates (Normalized [0.0 - 1.0]):")
    for box in result["normalized_boxes"][:5]:
        print(f"  Component #{box['id']}: norm_box={box['norm_box']}, pixels={box['box_pixels']}")
