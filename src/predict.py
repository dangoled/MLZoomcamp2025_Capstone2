import pickle
import pandas as pd
import numpy as np

class HealthPredictor:
    def __init__(self, model_path='models/model.pkl'):
        """Initialize the predictor with a trained model."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Get feature names
        self.numeric_features = ['Age', 'Fast_Food_Meals_Per_Week', 'Average_Daily_Calories',
                                 'BMI', 'Physical_Activity_Hours_Per_Week', 'Sleep_Hours_Per_Day',
                                 'Energy_Level_Score', 'Doctor_Visits_Per_Year']
        self.categorical_features = ['Gender']
        self.feature_names = self.numeric_features + ['Gender_Female', 'Gender_Male', 'Gender_Other']
    
    def predict(self, data):
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame or dict with required features
        
        Returns:
            numpy array of predictions
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure all required columns are present
        for feature in self.numeric_features + self.categorical_features:
            if feature not in data.columns:
                if feature in self.numeric_features:
                    data[feature] = 0
                else:
                    data[feature] = 'Unknown'
        
        # Make prediction
        prediction = self.model.predict(data)
        return prediction
    
    def predict_proba(self, data):
        """
        For regression, return confidence intervals.
        """
        predictions = self.predict(data)
        # Simple confidence interval (adjust based on your needs)
        lower_bound = predictions - 1.0
        upper_bound = predictions + 1.0
        return predictions, lower_bound, upper_bound

def example_usage():
    """Example of how to use the predictor."""
    predictor = HealthPredictor()
    
    # Example data
    example_person = {
        'Age': 35,
        'Gender': 'Male',
        'Fast_Food_Meals_Per_Week': 5,
        'Average_Daily_Calories': 2500,
        'BMI': 24.5,
        'Physical_Activity_Hours_Per_Week': 6,
        'Sleep_Hours_Per_Day': 7,
        'Energy_Level_Score': 6,
        'Doctor_Visits_Per_Year': 2,
        'Digestive_Issues': 'No'
    }
    
    prediction = predictor.predict(example_person)
    print(f"Predicted Health Score: {prediction[0]:.2f}")
    
    return prediction

if __name__ == "__main__":
    example_usage()