"""
Unit tests for Week 3: GPT-4o-mini API Integration and JSON Schema Enforcement.
"""

import pytest
import jsonschema
import json
from unittest.mock import MagicMock, patch
import openai

from src.schema import (
    UI_COMPONENT_JSON_SCHEMA,
    validate_ui_payload,
    ComponentItem,
    UIPageGeneration
)
from src.gpt_client import (
    GPTCodeGenerator,
    MissingAPIKeyError,
    SchemaValidationError,
    GPTAPIError
)

@pytest.fixture
def sample_wireframe_data():
    return {
        "original_shape": (600, 800),
        "target_shape": (800, 800),
        "total_components": 2,
        "normalized_boxes": [
            {
                "id": 1,
                "box_pixels": [50, 50, 700, 100],
                "norm_box": [0.0625, 0.0625, 0.9375, 0.1875],
                "norm_wh": [0.875, 0.125],
                "area_ratio": 0.109
            },
            {
                "id": 2,
                "box_pixels": [100, 200, 200, 60],
                "norm_box": [0.125, 0.25, 0.375, 0.325],
                "norm_wh": [0.25, 0.075],
                "area_ratio": 0.018
            }
        ]
    }

@pytest.fixture
def valid_payload():
    return {
        "page_title": "Login Page Wireframe",
        "layout_type": "flexbox",
        "components": [
            {
                "component_id": "comp-1",
                "component_type": "navbar",
                "bounding_box": [0.0, 0.0, 1.0, 0.1],
                "html": "<nav class=\"navbar\"><h1>App</h1></nav>",
                "css_classes": "navbar"
            },
            {
                "component_id": "comp-2",
                "component_type": "button",
                "bounding_box": [0.4, 0.5, 0.6, 0.58],
                "html": "<button class=\"btn-primary\">Submit</button>",
                "css_classes": "btn-primary"
            }
        ],
        "full_html": "<!DOCTYPE html><html><body><nav class=\"navbar\"><h1>App</h1></nav></body></html>",
        "full_css": "body { margin: 0; } .navbar { background: #222; }"
    }

# ============================================================================
# Task 1: API Key Management & Security Tests
# ============================================================================

def test_api_key_missing_raises_error():
    generator = GPTCodeGenerator(api_key=None)
    generator.api_key = ""
    with pytest.raises(MissingAPIKeyError):
        generator.get_client()

def test_api_key_placeholder_raises_error():
    generator = GPTCodeGenerator(api_key="your_openai_api_key_here")
    with pytest.raises(MissingAPIKeyError):
        generator.get_client()

def test_api_key_masking():
    generator = GPTCodeGenerator(api_key="sk-proj-1234567890abcdef123456")
    masked = generator.masked_api_key
    assert masked.startswith("sk-")
    assert masked.endswith("3456")
    assert "1234567890abcdef" not in masked

# ============================================================================
# Task 2 & 3: JSON Schema Enforcement Tests
# ============================================================================

def test_valid_payload_passes_schema(valid_payload):
    result = validate_ui_payload(valid_payload)
    assert result == valid_payload

def test_invalid_payload_missing_required_field(valid_payload):
    del valid_payload["full_html"]
    with pytest.raises(jsonschema.ValidationError):
        validate_ui_payload(valid_payload)

def test_invalid_component_type(valid_payload):
    valid_payload["components"][0]["component_type"] = "unsupported_unknown_widget"
    with pytest.raises(jsonschema.ValidationError):
        validate_ui_payload(valid_payload)

def test_unsupported_additional_property(valid_payload):
    valid_payload["unexpected_injected_field"] = "bad_data"
    with pytest.raises(jsonschema.ValidationError):
        validate_ui_payload(valid_payload)

def test_pydantic_type_validation(valid_payload):
    model = UIPageGeneration(**valid_payload)
    assert model.page_title == "Login Page Wireframe"
    assert len(model.components) == 2
    assert model.components[0].component_type == "navbar"

# ============================================================================
# Task 2: Prompt Construction Tests
# ============================================================================

def test_prompt_construction(sample_wireframe_data):
    generator = GPTCodeGenerator(api_key="sk-testkey123456")
    messages = generator.build_prompt(sample_wireframe_data)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "comp-1" in messages[1]["content"]

def test_multimodal_prompt_construction(sample_wireframe_data):
    generator = GPTCodeGenerator(api_key="sk-testkey123456")
    messages = generator.build_prompt(sample_wireframe_data, image_base64="iVBORw0KGgoAAAANSUhEUg==")
    assert isinstance(messages[1]["content"], list)
    assert messages[1]["content"][1]["type"] == "image_url"

# ============================================================================
# Task 4: API Response & Error Handling Tests
# ============================================================================

def test_mock_generation_structure(sample_wireframe_data):
    generator = GPTCodeGenerator()
    result = generator.generate_code(sample_wireframe_data, mock=True)
    assert "page_title" in result
    assert "layout_type" in result
    assert "components" in result
    assert "full_html" in result
    assert "full_css" in result
    # Verify mock response satisfies strict schema
    assert validate_ui_payload(result) is not None

def test_authentication_error_handling(sample_wireframe_data):
    generator = GPTCodeGenerator(api_key="sk-invalid-key-example")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
        "Incorrect API key provided", response=MagicMock(status_code=401), body={}
    )
    generator._client = mock_client

    with pytest.raises(GPTAPIError) as exc_info:
        generator.generate_code(sample_wireframe_data, mock=False)
    assert "Authentication failed" in str(exc_info.value)

def test_rate_limit_error_handling(sample_wireframe_data):
    generator = GPTCodeGenerator(api_key="sk-valid-looking-key123")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        "Rate limit reached", response=MagicMock(status_code=429), body={}
    )
    generator._client = mock_client

    with pytest.raises(GPTAPIError) as exc_info:
        generator.generate_code(sample_wireframe_data, mock=False)
    assert "rate limit exceeded" in str(exc_info.value)

def test_malformed_json_response_handling(sample_wireframe_data):
    generator = GPTCodeGenerator(api_key="sk-valid-key123")
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Invalid { Not JSON }"))]
    mock_client.chat.completions.create.return_value = mock_resp
    generator._client = mock_client

    with pytest.raises(SchemaValidationError):
        generator.generate_code(sample_wireframe_data, mock=False)
