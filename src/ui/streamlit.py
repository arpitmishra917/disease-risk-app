import streamlit as st
import requests

# Set page configuration
st.set_page_config(page_title="Disease Risk Predictor", layout="centered")

st.title("🏥 Disease Risk Prediction App")
st.write("Enter your health metrics below to assess your potential risk.")

# Create input form for the metrics
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        calories = st.number_input("Calories Consumed", min_value=0, value=2000)
        daily_steps = st.number_input("Daily Steps", min_value=0, value=5000)
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)

    with col2:
        cholesterol = st.number_input("Cholesterol", min_value=50, value=180)
        systolic_bp = st.number_input("Systolic BP", min_value=50, value=120)
        diastolic_bp = st.number_input("Diastolic BP", min_value=30, value=80)
        resting_hr = st.number_input("Resting Heart Rate", min_value=30, value=70)
        water_intake = st.number_input("Water Intake (Liters)", min_value=0.0, value=2.0)

    submit_button = st.form_submit_button(label="Predict Health Risk")

# Handle form submission
if submit_button:
    # Prepare the data payload matching your FastAPI Pydantic model
    payload = {
        "calories_consumed": float(calories),
        "daily_steps": float(daily_steps),
        "bmi": float(bmi),
        "cholesterol": float(cholesterol),
        "diastolic_bp": float(diastolic_bp),
        "systolic_bp": float(systolic_bp),
        "water_intake_l": float(water_intake),
        "sleep_hours": float(sleep_hours),
        "age": float(age),
        "resting_hr": float(resting_hr)
    }

    try:
        # Send POST request to FastAPI backend
        # Ensure your FastAPI server is running on this URL
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            status = result.get("Health status")
            
            if "not" in status.lower():
                st.success(f"✅ Result: {status}")
            else:
                st.error(f"⚠️ Result: {status}")
        else:
            st.error(f"Error: Received status code {response.status_code} from API.")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI server. Is it running?")
