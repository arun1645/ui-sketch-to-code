import sys
import subprocess
import requests

def check_python_deps():
    print("=== Checking Python Dependencies ===")
    try:
        import cv2
        print(f"[OK] OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"[FAIL] OpenCV import failed: {e}")
        return False

    try:
        import numpy as np
        print(f"[OK] NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"[FAIL] NumPy import failed: {e}")
        return False

    try:
        import PIL
        print(f"[OK] Pillow version: {PIL.__version__}")
    except ImportError as e:
        print(f"[FAIL] Pillow import failed: {e}")
        return False

    # Test OpenCV functionality with synthetic image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[OK] OpenCV Contour test detected {len(contours)} contour(s)")
    return True

def check_ollama():
    print("\n=== Checking Ollama Connection & Model Status ===")
    url = "http://localhost:11434/api/tags"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            models = res.json().get("models", [])
            model_names = [m.get("name") for m in models]
            print(f"[OK] Ollama API reachable. Installed models: {model_names}")
            has_llava = any("llava" in m.lower() for m in model_names)
            if has_llava:
                print("[OK] LLaVA model available locally!")
            else:
                print("[WARNING] LLaVA model not found in Ollama tags yet.")
            return True
        else:
            print(f"[FAIL] Ollama HTTP status: {res.status_code}")
            return False
    except Exception as e:
        print(f"[INFO] Ollama service connection note: {e}")
        print("       (Ollama server can be started via 'ollama serve')")
        return False

if __name__ == "__main__":
    py_ok = check_python_deps()
    ollama_ok = check_ollama()
    print("\n=== Verification Summary ===")
    print(f"Python Core Stack: {'PASS' if py_ok else 'FAIL'}")
    print(f"Ollama LLaVA API:  {'PASS' if ollama_ok else 'WARN/FAIL'}")
