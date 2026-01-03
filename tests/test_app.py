import pytest
from fastapi.testclient import TestClient
# Import the 'app' object from your main script (assuming it's named main.py)
from src.api.fastapi_app import app

client = TestClient(app)

@pytest.fixture
def sample_payload():
    return {
        "calories_consumed": 125.0,
        "daily_steps": 3000.0,
        "bmi": 25.0,
        "cholesterol": 120.0,
        "diastolic_bp": 160.0,
        "systolic_bp": 140.0,
        "water_intake_l": 90.0,
        "sleep_hours": 8.0,
        "age": 30.0,
        "resting_hr": 95.0
    }

def test_read_root():
    """Tests the GET / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "welcome to Disease Risk prediction App"}

def test_predict_success(sample_payload):
    """Tests successful prediction with valid data."""
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "Health status" in data
    # Matches the exact strings in your FastAPI logic
    assert data["Health status"] in [
        "Your health is under danger", 
        "Your health is not in under danger"
    ]

def test_predict_zero_division_safety():
    """Verifies that 0 values for calories/sleep don't crash the server."""
    zero_payload = {
        "calories_consumed": 0, # Tests hydration_index logic
        "daily_steps": 100,
        "bmi": 20,
        "cholesterol": 100,
        "diastolic_bp": 80,
        "systolic_bp": 120,
        "water_intake_l": 2,
        "sleep_hours": 0,       # Tests activity_sleep_ratio logic
        "age": 25,
        "resting_hr": 70
    }
    response = client.post("/predict", json=zero_payload)
    assert response.status_code == 200
    assert "Health status" in response.json()

def test_predict_invalid_input():
    """Tests Pydantic validation (e.g., passing a string instead of a float)."""
    bad_payload = {"calories_consumed": "high"}
    response = client.post("/predict", json=bad_payload)
    # FastAPI returns 422 for validation errors
    assert response.status_code == 422 

def test_predict_missing_field():
    """Tests that a missing required field triggers an error."""
    incomplete_payload = {"calories_consumed": 100.0}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422
