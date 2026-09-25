import cv2
import numpy as np
import os
from typing import Union, List, Dict, Tuple, Optional

class WireframePreprocessor:
    """
    OpenCV Pre-processing pipeline for UI Sketch & Wireframe images.
    Enhances image contrast, reduces noise, normalizes bounding box coordinates,
    and formats wireframes for downstream LLM code generation.
    """

    def __init__(self, target_size: Tuple[int, int] = (800, 800), min_contour_area: int = 150):
        self.target_size = target_size
        self.min_contour_area = min_contour_area
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def load_image(self, input_source: Union[str, np.ndarray]) -> np.ndarray:
        """Loads image from file path or validates existing numpy array."""
        if isinstance(input_source, str):
            if not os.path.exists(input_source):
                raise FileNotFoundError(f"Input image file not found: {input_source}")
            img = cv2.imread(input_source)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {input_source}")
            return img
        elif isinstance(input_source, np.ndarray):
            return input_source.copy()
        else:
            raise TypeError("Input source must be a file path string or numpy ndarray.")

    def enhance_contrast(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhances contrast using CLAHE and produces a clean binarized image.
        Returns tuple of (enhanced_grayscale, binary_thresholded).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        enhanced_gray = self.clahe.apply(gray)

        # 2. Denoise with Bilateral Filter to smooth noise while preserving sharp sketch edges
        denoised = cv2.bilateralFilter(enhanced_gray, d=5, sigmaColor=50, sigmaSpace=50)

        # 3. Adaptive Gaussian Thresholding for clean binarization
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        return enhanced_gray, binary

    def resize_and_pad(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resizes image to target_size preserving aspect ratio with white background padding.
        Returns (padded_image, scale_factor, (top_pad, left_pad)).
        """
        target_w, target_h = self.target_size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        top_pad = (target_h - new_h) // 2
        bottom_pad = target_h - new_h - top_pad
        left_pad = (target_w - new_w) // 2
        right_pad = target_w - new_w - left_pad

        if len(image.shape) == 3:
            padded = cv2.copyMakeBorder(
                resized, top_pad, bottom_pad, left_pad, right_pad,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        else:
            padded = cv2.copyMakeBorder(
                resized, top_pad, bottom_pad, left_pad, right_pad,
                cv2.BORDER_CONSTANT, value=0 if len(np.unique(image)) <= 2 else 255
            )

        return padded, scale, (top_pad, left_pad)

    def extract_and_normalize_boxes(
        self, binary_image: np.ndarray, original_shape: Tuple[int, int]
    ) -> List[Dict[str, Union[int, List[float], float]]]:
        """
        Extracts contour bounding boxes and normalizes coordinates to [0.0, 1.0] relative scale.
        """
        orig_h, orig_w = original_shape[:2]
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        normalized_boxes = []
        box_id = 1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Skip full canvas bounding box
            if w >= binary_image.shape[1] * 0.98 and h >= binary_image.shape[0] * 0.98:
                continue

            # Calculate normalized relative coordinates [0.0, 1.0]
            norm_xmin = round(float(x) / binary_image.shape[1], 4)
            norm_ymin = round(float(y) / binary_image.shape[0], 4)
            norm_xmax = round(float(x + w) / binary_image.shape[1], 4)
            norm_ymax = round(float(y + h) / binary_image.shape[0], 4)
            norm_width = round(float(w) / binary_image.shape[1], 4)
            norm_height = round(float(h) / binary_image.shape[0], 4)

            normalized_boxes.append({
                "id": box_id,
                "box_pixels": [x, y, w, h],
                "norm_box": [norm_xmin, norm_ymin, norm_xmax, norm_ymax],
                "norm_wh": [norm_width, norm_height],
                "area_ratio": round(float(area) / (binary_image.shape[0] * binary_image.shape[1]), 5)
            })
            box_id += 1

        # Sort boxes by Y top coordinate then X left coordinate
        normalized_boxes.sort(key=lambda b: (b["norm_box"][1], b["norm_box"][0]))
        return normalized_boxes

    def process(
        self, input_source: Union[str, np.ndarray], output_dir: Optional[str] = None
    ) -> Dict[str, Union[np.ndarray, List[Dict], Tuple[int, int]]]:
        """
        Executes full pre-processing pipeline:
        1. Loads image
        2. Enhances contrast and binarizes
        3. Resizes and pads to standardized target dimensions
        4. Extracts and normalizes bounding box coordinates
        5. Saves processed outputs if output_dir is provided
        """
        orig_img = self.load_image(input_source)
        orig_h, orig_w = orig_img.shape[:2]

        enhanced_gray, binary_inv = self.enhance_contrast(orig_img)
        padded_enhanced, _, _ = self.resize_and_pad(enhanced_gray)
        padded_binary, _, _ = self.resize_and_pad(binary_inv)

        normalized_boxes = self.extract_and_normalize_boxes(padded_binary, (orig_h, orig_w))

        # Create visualization image with bounding boxes drawn
        vis_img = cv2.cvtColor(padded_enhanced, cv2.COLOR_GRAY2BGR)
        for b in normalized_boxes:
            x, y, w, h = b["box_pixels"]
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                vis_img, f"#{b['id']}", (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
            )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "enhanced.png"), padded_enhanced)
            cv2.imwrite(os.path.join(output_dir, "binary.png"), padded_binary)
            cv2.imwrite(os.path.join(output_dir, "bounding_boxes.png"), vis_img)

        return {
            "original_shape": (orig_h, orig_w),
            "target_shape": self.target_size,
            "enhanced_image": padded_enhanced,
            "binary_image": padded_binary,
            "visualization_image": vis_img,
            "normalized_boxes": normalized_boxes,
            "total_components": len(normalized_boxes)
        }
