from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
import numpy as np
from typing import List
import logging
from datetime import datetime

from src.schemas import (
    HealthData, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthRecommendation
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fast Food Health Impact Prediction API",
    description="API for predicting health scores based on lifestyle factors",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Model:
    """Model wrapper for loading and using the trained model."""
    def __init__(self):
        self.model = None
        self.loaded = False
    
    def load_model(self):
        """Load the model from disk."""
        try:
            with open('models/model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            self.loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the loaded model."""
        if not self.loaded:
            self.load_model()
        return self.model.predict(data)

# Initialize model wrapper
model_wrapper = Model()

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        model_wrapper.load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fast Food Health Impact Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health_check": "/health",
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "recommendations": "/recommendations",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_wrapper.loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_health(data: HealthData):
    """
    Predict health score for a single individual.
    
    - **Age**: Age in years (18-100)
    - **Gender**: Male, Female, or Other
    - **Fast_Food_Meals_Per_Week**: Number of fast food meals per week
    - **Average_Daily_Calories**: Estimated daily calorie intake
    - **BMI**: Body Mass Index
    - **Physical_Activity_Hours_Per_Week**: Weekly exercise hours
    - **Sleep_Hours_Per_Day**: Average daily sleep hours
    - **Energy_Level_Score**: Self-reported energy level (1-10)
    - **Doctor_Visits_Per_Year**: Annual doctor visits
    - **Digestive_Issues**: Yes or No
    """
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Make prediction
        prediction = model_wrapper.predict(input_data)
        
        # Create response
        response = PredictionResponse.from_prediction(float(prediction[0]))
        
        logger.info(f"Prediction made: {response.health_score}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Make predictions for multiple individuals."""
    try:
        # Convert input to DataFrame
        data_dicts = [item.dict() for item in request.data]
        input_data = pd.DataFrame(data_dicts)
        
        # Make predictions
        predictions = model_wrapper.predict(input_data)
        
        # Create individual responses
        prediction_responses = [
            PredictionResponse.from_prediction(float(pred))
            for pred in predictions
        ]
        
        # Calculate average
        avg_score = float(np.mean(predictions))
        
        # Generate recommendations
        recommendations = generate_recommendations(input_data, predictions)
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            average_score=round(avg_score, 2),
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations")
async def get_recommendations(data: HealthData):
    """Get personalized health recommendations."""
    try:
        # Make prediction first
        input_data = pd.DataFrame([data.dict()])
        prediction = model_wrapper.predict(input_data)[0]
        
        # Generate recommendations
        recommendations = generate_personalized_recommendations(
            data.dict(), prediction
        )
        
        return {
            "predicted_score": round(float(prediction), 2),
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_recommendations(data: pd.DataFrame, predictions: np.ndarray) -> List[str]:
    """Generate general recommendations based on predictions."""
    recommendations = []
    avg_prediction = np.mean(predictions)
    
    if avg_prediction < 5:
        recommendations.append("Overall health scores are low. Consider lifestyle improvements.")
    
    # Check fast food consumption
    if data['Fast_Food_Meals_Per_Week'].mean() > 7:
        recommendations.append("High fast food consumption detected. Consider reducing to 3-4 meals per week.")
    
    # Check physical activity
    if data['Physical_Activity_Hours_Per_Week'].mean() < 3:
        recommendations.append("Low physical activity. Aim for at least 150 minutes per week.")
    
    # Check sleep
    if data['Sleep_Hours_Per_Day'].mean() < 6:
        recommendations.append("Insufficient sleep. Aim for 7-9 hours per night.")
    
    return recommendations[:5]  # Return top 5 recommendations

def generate_personalized_recommendations(data: dict, prediction: float) -> List[dict]:
    """Generate personalized recommendations."""
    recommendations = []
    
    # Fast food recommendation
    if data['Fast_Food_Meals_Per_Week'] > 5:
        recommendations.append({
            "category": "Nutrition",
            "message": f"Reduce fast food consumption from {data['Fast_Food_Meals_Per_Week']} to 3-4 meals per week",
            "priority": "high" if data['Fast_Food_Meals_Per_Week'] > 8 else "medium",
            "impact": "Reduces health risks and improves energy levels"
        })
    
    # Physical activity recommendation
    if data['Physical_Activity_Hours_Per_Week'] < 3:
        recommendations.append({
            "category": "Exercise",
            "message": "Increase physical activity to at least 150 minutes per week",
            "priority": "high" if data['Physical_Activity_Hours_Per_Week'] < 1 else "medium",
            "impact": "Improves cardiovascular health and boosts energy"
        })
    
    # Sleep recommendation
    if data['Sleep_Hours_Per_Day'] < 6:
        recommendations.append({
            "category": "Sleep",
            "message": "Aim for 7-9 hours of sleep per night",
            "priority": "high" if data['Sleep_Hours_Per_Day'] < 5 else "medium",
            "impact": "Improves cognitive function and overall well-being"
        })
    
    # BMI recommendation
    if data['BMI'] > 25:
        recommendations.append({
            "category": "Weight Management",
            "message": "Consider weight management strategies",
            "priority": "high" if data['BMI'] > 30 else "medium",
            "impact": "Reduces risk of chronic diseases"
        })
    
    # Energy level recommendation
    if data['Energy_Level_Score'] < 5:
        recommendations.append({
            "category": "Energy",
            "message": "Low energy levels detected. Consider dietary and exercise improvements",
            "priority": "medium",
            "impact": "Improves daily productivity and mood"
        })
    
    return recommendations

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)