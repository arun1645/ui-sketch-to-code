"""
Demonstration and verification script for Week 3: GPT-4o-mini API Integration.
Executes schema validation, prompt building, and code generation.
"""

import os
import sys
import json

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import WireframePreprocessor
from src.gpt_client import GPTCodeGenerator, MissingAPIKeyError
from src.schema import validate_ui_payload, UI_COMPONENT_JSON_SCHEMA

def main():
    print("=" * 65)
    print("  Week 3: GPT-4o-mini API Integration Verification")
    print("=" * 65)

    # 1. API Key & Security Check
    generator = GPTCodeGenerator()
    print(f"\n[1] API Key Management:")
    print(f"    - Target Model: {generator.model}")
    print(f"    - API Key Configured: {'Yes' if generator.api_key and 'your_openai' not in generator.api_key else 'No (using secure demo/mock mode)'}")
    print(f"    - Masked Key: {generator.masked_api_key}")
    print(f"    - .env File Status: {'.env file exists' if os.path.exists('.env') else '.env not created yet (using .env.example template)'}")

    # 2. Wireframe Pre-Processing (Connecting Week 2 -> Week 3)
    sample_img = "tests/sample_wireframe.png"
    if not os.path.exists(sample_img):
        print(f"\n[!] Generating synthetic wireframe fixture...")
        from tests.generate_samples import create_synthetic_wireframe
        create_synthetic_wireframe(sample_img)

    print(f"\n[2] Connecting Pre-processor to Code Generator:")
    preprocessor = WireframePreprocessor()
    wireframe_result = preprocessor.process(sample_img)
    print(f"    - Original Image Shape: {wireframe_result['original_shape']}")
    print(f"    - Detected Component Count: {wireframe_result['total_components']}")

    # 3. JSON Schema Output Enforcement
    print(f"\n[3] JSON Schema Enforcement Definition:")
    print(f"    - Schema Name: {UI_COMPONENT_JSON_SCHEMA['name']}")
    print(f"    - Required Fields: {UI_COMPONENT_JSON_SCHEMA['schema']['required']}")
    print(f"    - Strict Mode: {UI_COMPONENT_JSON_SCHEMA['strict']}")

    # 4. Generate Code
    print(f"\n[4] Generating UI Component Code:")
    can_call_live = bool(generator.api_key and "your_openai" not in generator.api_key and generator.api_key.startswith("sk-"))

    if can_call_live:
        print("    - Mode: LIVE OpenAI API (GPT-4o-mini)")
        try:
            result = generator.generate_code(wireframe_result, mock=False)
        except Exception as e:
            print(f"    - Live API error ({e}). Falling back to schema-validated mock.")
            result = generator.generate_code(wireframe_result, mock=True)
    else:
        print("    - Mode: SCHEMA-ENFORCED VERIFICATION (Mock Engine)")
        result = generator.generate_code(wireframe_result, mock=True)

    # 5. Schema Validation Check
    validate_ui_payload(result)
    print(f"    [OK] Payload successfully validated against JSON schema!")
    print(f"    - Page Title: {result['page_title']}")
    print(f"    - Layout Type: {result['layout_type']}")
    print(f"    - Generated Components Count: {len(result['components'])}")

    # 6. Save generated HTML and CSS
    out_dir = "output/week3"
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, "index.html")
    css_path = os.path.join(out_dir, "styles.css")
    json_path = os.path.join(out_dir, "payload.json")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(result["full_html"])
    with open(css_path, "w", encoding="utf-8") as f:
        f.write(result["full_css"])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[5] Artifacts Generated:")
    print(f"    - HTML File: {html_path}")
    print(f"    - CSS File: {css_path}")
    print(f"    - JSON Payload: {json_path}")
    print("\n" + "=" * 65)
    print("  Week 3 Verification Completed Successfully!")
    print("=" * 65)

if __name__ == "__main__":
    main()
