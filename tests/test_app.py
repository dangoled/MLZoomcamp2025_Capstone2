import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint():
    """Test prediction endpoint with valid data."""
    test_data = {
        "Age": 35,
        "Gender": "Male",
        "Fast_Food_Meals_Per_Week": 5,
        "Average_Daily_Calories": 2500,
        "BMI": 24.5,
        "Physical_Activity_Hours_Per_Week": 6,
        "Sleep_Hours_Per_Day": 7,
        "Energy_Level_Score": 6,
        "Doctor_Visits_Per_Year": 2,
        "Digestive_Issues": "No"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "health_score" in data
    assert "health_category" in data
    assert 1 <= data["health_score"] <= 10

def test_predict_invalid_data():
    """Test prediction endpoint with invalid data."""
    test_data = {
        "Age": 150,  # Invalid age
        "Gender": "Male",
        "Fast_Food_Meals_Per_Week": 5,
        "Average_Daily_Calories": 2500,
        "BMI": 24.5,
        "Physical_Activity_Hours_Per_Week": 6,
        "Sleep_Hours_Per_Day": 7,
        "Energy_Level_Score": 6,
        "Doctor_Visits_Per_Year": 2,
        "Digestive_Issues": "No"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error

def test_batch_predict():
    """Test batch prediction endpoint."""
    test_data = {
        "data": [
            {
                "Age": 35,
                "Gender": "Male",
                "Fast_Food_Meals_Per_Week": 5,
                "Average_Daily_Calories": 2500,
                "BMI": 24.5,
                "Physical_Activity_Hours_Per_Week": 6,
                "Sleep_Hours_Per_Day": 7,
                "Energy_Level_Score": 6,
                "Doctor_Visits_Per_Year": 2,
                "Digestive_Issues": "No"
            },
            {
                "Age": 45,
                "Gender": "Female",
                "Fast_Food_Meals_Per_Week": 2,
                "Average_Daily_Calories": 2000,
                "BMI": 22.0,
                "Physical_Activity_Hours_Per_Week": 8,
                "Sleep_Hours_Per_Day": 8,
                "Energy_Level_Score": 8,
                "Doctor_Visits_Per_Year": 1,
                "Digestive_Issues": "No"
            }
        ]
    }
    
    response = client.post("/predict/batch", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert "average_score" in data

def test_recommendations_endpoint():
    """Test recommendations endpoint."""
    test_data = {
        "Age": 35,
        "Gender": "Male",
        "Fast_Food_Meals_Per_Week": 10,  # High fast food consumption
        "Average_Daily_Calories": 3000,
        "BMI": 28.0,  # Overweight
        "Physical_Activity_Hours_Per_Week": 1,  # Low activity
        "Sleep_Hours_Per_Day": 5,  # Low sleep
        "Energy_Level_Score": 3,
        "Doctor_Visits_Per_Year": 4,
        "Digestive_Issues": "Yes"
    }
    
    response = client.post("/recommendations", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_score" in data
    assert "recommendations" in data