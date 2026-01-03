import numpy as np
from fastapi import FastAPI,HTTPException
import logging
# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from pydantic import BaseModel
import joblib


def load_model(file_path: str):
    
    try:
        with open(file_path, 'rb') as file:
            model = joblib.load(file)
        logger.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

class InputData(BaseModel):
    calories_consumed:float
    daily_steps:float
    bmi:float
    cholesterol:float
    diastolic_bp:float
    systolic_bp:float
    water_intake_l:float
    sleep_hours:float
    age:float
    resting_hr:float


app=FastAPI(title="Disease Risk prediction App")

model=load_model('./models/logistic_regression.pkl')

@app.get("/")
def read_root():
    return {"message":"welcome to Disease Risk prediction App"}

@app.post("/predict")
def predict(data:InputData):
    
    hydration_index=(data.water_intake_l/data.calories_consumed if data.calories_consumed>0 else 0)

    activity_sleep_ratio=(data.daily_steps/data.sleep_hours if data.sleep_hours>0 else 0)

    age_cholesterol=data.age*data.cholesterol

    resting_bp_product=data.resting_hr*data.systolic_bp

    if model is None:
        raise HTTPException(status_code=500,detail="model not found")
    
    try:
        input_array=np.array([[
            data.calories_consumed,
            data.daily_steps,
            data.bmi,
            hydration_index,
            activity_sleep_ratio,
            age_cholesterol,
            resting_bp_product,
            data.cholesterol,
            data.diastolic_bp,
            data.systolic_bp]])
        
        thresh=0.51
        health_status=""
        y_proba = model.predict_proba(input_array)[0][1]
        if y_proba >= thresh:
            health_status= "Your health is under danger"
        else:
            health_status= "Your health is not in under danger"

        return {"Health status":health_status}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
    raise HTTPException(status_code=400, detail="Prediction error")
    

