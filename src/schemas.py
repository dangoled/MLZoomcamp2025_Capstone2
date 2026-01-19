from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
import numpy as np

class HealthData(BaseModel):
    """Schema for health prediction input data."""
    Age: int = Field(ge=18, le=100, description="Age between 18 and 100")
    Gender: Literal['Male', 'Female', 'Other'] = Field(description="Gender")
    Fast_Food_Meals_Per_Week: int = Field(ge=0, le=20, description="Fast food meals per week")
    Average_Daily_Calories: float = Field(ge=1000, le=5000, description="Average daily calories")
    BMI: float = Field(ge=15, le=50, description="Body Mass Index")
    Physical_Activity_Hours_Per_Week: float = Field(ge=0, le=40, description="Physical activity hours per week")
    Sleep_Hours_Per_Day: float = Field(ge=3, le=12, description="Sleep hours per day")
    Energy_Level_Score: int = Field(ge=1, le=10, description="Energy level score (1-10)")
    Doctor_Visits_Per_Year: int = Field(ge=0, le=20, description="Doctor visits per year")
    Digestive_Issues: Literal['Yes', 'No'] = Field(description="Digestive issues")
    
    @validator('BMI')
    def validate_bmi(cls, v):
        if v < 10 or v > 60:
            raise ValueError('BMI must be between 10 and 60')
        return v
    
    @validator('Energy_Level_Score')
    def validate_energy_score(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Energy level score must be between 1 and 10')
        return v

class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    health_score: float = Field(description="Predicted overall health score (1-10)")
    confidence_interval_lower: float = Field(description="Lower bound of 95% confidence interval")
    confidence_interval_upper: float = Field(description="Upper bound of 95% confidence interval")
    health_category: str = Field(description="Health category based on score")
    
    @classmethod
    def from_prediction(cls, prediction: float):
        """Create response from prediction."""
        # Simple confidence interval
        lower = max(1, prediction - 1.0)
        upper = min(10, prediction + 1.0)
        
        # Categorize health score
        if prediction >= 8:
            category = "Excellent"
        elif prediction >= 6:
            category = "Good"
        elif prediction >= 4:
            category = "Fair"
        else:
            category = "Poor"
        
        return cls(
            health_score=round(prediction, 2),
            confidence_interval_lower=round(lower, 2),
            confidence_interval_upper=round(upper, 2),
            health_category=category
        )

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction."""
    data: list[HealthData]

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    predictions: list[PredictionResponse]
    average_score: float
    recommendations: list[str]

class HealthRecommendation(BaseModel):
    """Schema for health recommendations."""
    category: str
    message: str
    priority: Literal['high', 'medium', 'low']