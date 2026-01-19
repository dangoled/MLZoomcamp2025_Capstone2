import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df

def prepare_data(df):
    """Prepare features and target."""
    X = df.drop('Overall_Health_Score', axis=1)
    y = df['Overall_Health_Score']
    return X, y

def create_preprocessor():
    """Create preprocessing pipeline."""
    numeric_features = ['Age', 'Fast_Food_Meals_Per_Week', 'Average_Daily_Calories',
                       'BMI', 'Physical_Activity_Hours_Per_Week', 'Sleep_Hours_Per_Day',
                       'Energy_Level_Score', 'Doctor_Visits_Per_Year']
    categorical_features = ['Gender']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
             categorical_features)
        ]
    )
    return preprocessor

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and evaluate them."""
    preprocessor = create_preprocessor()
    
    models = {
        'linear_regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ]),
        'random_forest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ]),
        'gradient_boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'metrics': {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        }
        
        print(f"  R²: {results[name]['metrics']['r2']:.4f}")
        print(f"  RMSE: {results[name]['metrics']['rmse']:.4f}")
    
    return results

def save_best_model(results, output_path='models/'):
    """Save the best model based on R² score."""
    best_model_name = max(results.keys(), 
                          key=lambda x: results[x]['metrics']['r2'])
    best_model = results[best_model_name]['model']
    
    # Save the model
    model_path = f"{output_path}model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save metrics
    metrics = {}
    for name, result in results.items():
        metrics[name] = result['metrics']
    
    metrics_path = f"{output_path}metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nBest model: {best_model_name}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    
    return model_path, metrics_path

def main():
    """Main training function."""
    print("Starting model training...")
    
    # Load data
    df = load_data('data/fast_food_consumption_health_impact_dataset.csv')
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Prepare data
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5)
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train models
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Save best model
    save_best_model(results, output_path='models/')
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()