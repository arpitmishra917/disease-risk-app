import pytest
import joblib
import pandas as pd
from pydantic import BaseModel
import numpy as np


@pytest.fixture
def load_model():
    # Note: This requires the model file "models/logistic_regression.pkl" to exist at the specified path.
    return joblib.load("models/logistic_regression.pkl")


@pytest.fixture
def sample_input():
    calories_consumed=125
    daily_steps=3000
    bmi=25
    cholesterol=120
    diastolic_bp=160
    systolic_bp=140
    water_intake_l=90
    sleep_hours=8
    age=30
    resting_hr=95

    # Feature engineering as done for model training
    hydration_index=(water_intake_l/calories_consumed if calories_consumed>0 else 0)

    activity_sleep_ratio=(daily_steps/sleep_hours if sleep_hours>0 else 0)

    age_cholesterol=age*cholesterol

    resting_bp_product=resting_hr*systolic_bp

    # Return a single-row DataFrame with all required input features
    return pd.DataFrame({
            "calories_consumed":[calories_consumed], # Wrap values in list for DataFrame
            "daily_steps":[daily_steps],
            "bmi":[bmi],
            "hydration_index":[hydration_index],
            "activity_sleep_ratio":[activity_sleep_ratio],
            "age_cholesterol":[age_cholesterol],
            "resting_bp_product":[resting_bp_product],
            "cholesterol":[cholesterol],
            "diastolic_bp":[diastolic_bp],
            "systolic_bp":[systolic_bp]
    })





def test_prediction_probabilities(load_model, sample_input):
    """Ensure predicted probabilities are between 0 and 1 and sum to 1."""
    probs = load_model.predict_proba(sample_input)
    # Check bounds
    assert np.all(probs >= 0) and np.all(probs <= 1)
    # Check sum for each sample across classes is approximately 1
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_output_shape(load_model, sample_input):
    """Verify the model returns exactly one prediction per input sample."""
    predictions = load_model.predict(sample_input)
    assert len(predictions) == len(sample_input)


def test_prediction_consistency(load_model, sample_input):
    """Ensure the model produces the same output for identical inputs."""
    pred1 = load_model.predict_proba(sample_input)
    pred2 = load_model.predict_proba(sample_input)
    np.testing.assert_array_almost_equal(pred1, pred2)
