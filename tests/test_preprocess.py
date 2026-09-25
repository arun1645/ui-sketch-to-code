import os
import sys
import pytest
import numpy as np
import cv2

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import WireframePreprocessor

@pytest.fixture
def preprocessor():
    return WireframePreprocessor(target_size=(800, 800), min_contour_area=100)

@pytest.fixture
def synthetic_image():
    img = np.full((600, 800, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (750, 150), (30, 30, 30), 3) # Header box
    cv2.rectangle(img, (100, 200), (350, 500), (40, 40, 40), 3) # Button/Card box
    return img

def test_load_image(preprocessor, synthetic_image, tmp_path):
    # Test numpy array loading
    img = preprocessor.load_image(synthetic_image)
    assert isinstance(img, np.ndarray)
    assert img.shape == (600, 800, 3)

    # Test file path loading
    temp_file = str(tmp_path / "test.png")
    cv2.imwrite(temp_file, synthetic_image)
    img_from_file = preprocessor.load_image(temp_file)
    assert isinstance(img_from_file, np.ndarray)

    # Test non-existent file error
    with pytest.raises(FileNotFoundError):
        preprocessor.load_image("non_existent_path.png")

def test_contrast_enhancement(preprocessor, synthetic_image):
    enhanced, binary = preprocessor.enhance_contrast(synthetic_image)
    assert enhanced.shape[:2] == (600, 800)
    assert binary.shape[:2] == (600, 800)
    assert len(enhanced.shape) == 2  # Grayscale
    assert len(binary.shape) == 2    # Binarized

def test_resize_and_pad(preprocessor, synthetic_image):
    padded, scale, (top, left) = preprocessor.resize_and_pad(synthetic_image)
    assert padded.shape[:2] == (800, 800)
    assert scale <= 1.0
    assert top >= 0 and left >= 0

def test_normalized_bounding_boxes(preprocessor, synthetic_image):
    result = preprocessor.process(synthetic_image)
    boxes = result["normalized_boxes"]

    assert isinstance(boxes, list)
    assert len(boxes) > 0

    for b in boxes:
        assert "id" in b
        assert "norm_box" in b
        assert "norm_wh" in b
        xmin, ymin, xmax, ymax = b["norm_box"]
        # Verify coordinates strictly bounded between 0.0 and 1.0
        assert 0.0 <= xmin <= 1.0
        assert 0.0 <= ymin <= 1.0
        assert 0.0 <= xmax <= 1.0
        assert 0.0 <= ymax <= 1.0
        assert xmin <= xmax
        assert ymin <= ymax

def test_full_pipeline_structure(preprocessor, synthetic_image, tmp_path):
    out_dir = str(tmp_path / "outputs")
    res = preprocessor.process(synthetic_image, output_dir=out_dir)

    assert "original_shape" in res
    assert "target_shape" in res
    assert "enhanced_image" in res
    assert "binary_image" in res
    assert "visualization_image" in res
    assert "normalized_boxes" in res
    assert "total_components" in res

    assert os.path.exists(os.path.join(out_dir, "enhanced.png"))
    assert os.path.exists(os.path.join(out_dir, "binary.png"))
    assert os.path.exists(os.path.join(out_dir, "bounding_boxes.png"))
