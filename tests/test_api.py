import io
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the FastAPI app from your main API script
from src.api.main import app

client = TestClient(app)


def test_health_check():
    """Verify the API boots and the health endpoint responds."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "vlm_connected": True}


@patch("src.api.main.client.chat.completions.create")
@patch("src.api.main.retrieve_regulation")
def test_audit_compliance_logic(mock_retrieve_regulation, mock_vlm_create):
    """
    Test the Rule Engine's deterministic PASS/FAIL logic by mocking
    the VLM's JSON output and the RAG database retrieval.
    """
    # 1. Mock the RAG Database Retrieval
    # We force the database to return a strict 5.0 degree threshold for roof heat loss.
    mock_retrieve_regulation.return_value = {
        "max_allowed_delta_t": 5.0,
        "code_ref": "Part L - 4.1 Mock"
    }

    # 2. Mock the VLM Output
    # We inject two defects: one that passes (4.0 <= 5.0) and one that fails (7.5 > 5.0)
    mock_vlm_response = {
        "defects": [
            {
                "defect_type": "roof_heat_loss",
                "delta_t": 4.0,
                "bounding_box": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2}
            },
            {
                "defect_type": "roof_heat_loss",
                "delta_t": 7.5,
                "bounding_box": {"x_min": 0.5, "y_min": 0.5, "x_max": 0.6, "y_max": 0.6}
            }
        ]
    }

    # Construct the deep mock object to match OpenAI's Python client structure
    mock_message = MagicMock()
    mock_message.content = json.dumps(mock_vlm_response)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_vlm_create.return_value = mock_completion

    # 3. Create dummy images to satisfy the FastAPI File requirements
    dummy_rgb = io.BytesIO(b"dummy_rgb_data")
    dummy_thermal = io.BytesIO(b"dummy_thermal_data")

    # 4. Execute the API Request
    response = client.post(
        "/audit",
        files={
            "rgb_image": ("rgb.jpg", dummy_rgb, "image/jpeg"),
            "thermal_image": ("thermal.jpg", dummy_thermal, "image/jpeg")
        }
    )

    # 5. Assertions (The Rule Engine Validation)
    assert response.status_code == 200
    results = response.json().get("audit_results")

    assert len(results) == 2

    # First defect (4.0 delta) should PASS against the 5.0 threshold
    assert results[0]["detected_delta_t"] == 4.0
    assert results[0]["compliance_status"] == "PASS"

    # Second defect (7.5 delta) should FAIL against the 5.0 threshold
    assert results[1]["detected_delta_t"] == 7.5
    assert results[1]["compliance_status"] == "FAIL"