# MLZoomcamp2025_Capstone2 Project

## Fast Food Health Impact Prediction
 
### Problem Description
This is a health impact prediction project that predicts a person's overall health score based on their fast food consumption patterns, lifestyle factors, and demographic information. 

It uses the "fast_food_consumption_health_impact_dataset" from kaggle (https://www.kaggle.com/datasets/prince7489/fast-food-consumption-and-health-impact-dataset)

Relevance: This is a relevant real-world problem that can help:
- Healthcare professionals identify at-risk patients
- Individuals understand potential health impacts of their lifestyle choices
- Public health organizations target interventions
- Insurance companies assess risk profiles

The target variable is Overall_Health_Score (scale 1-10).

### Project Structure
```
fastfood-health-prediction/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── schemas.py
│   └── app.py
├── models/
│   └── (saved models)
├── data/
│   └── fast_food_consumption_health_impact_dataset.csv
├── tests/
│   └── test_app.py
├── Dockerfile
├── fly.toml
├── requirements.in
├── pyproject.toml
```

## Instructions

### 1. Install dependencies with UV:
 - Install uv if not installed
 - Create virtual environment <br>
uv venv .venv
 - Install dependencies <br>
uv pip install -r requirements.txt
### 2. Run the EDA notebook:
- jupyter notebook notebooks/exploration.ipynb
### 3. Train the model:
- python src/train.py
### 4. Run the API:
- uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
### 5. Test the API:
- Test prediction endpoint <br>
curl -X POST "http://localhost:8000/predict" \ <br>
  -H "Content-Type: application/json" \ <br>
  -d '{ <br>
    "Age": 35, <br>
    "Gender": "Male", <br>
    "Fast_Food_Meals_Per_Week": 5, <br>
    "Average_Daily_Calories": 2500, <br>
    "BMI": 24.5, <br>
    "Physical_Activity_Hours_Per_Week": 6, <br>
    "Sleep_Hours_Per_Day": 7, <br>
    "Energy_Level_Score": 6, <br>
    "Doctor_Visits_Per_Year": 2, <br>
    "Digestive_Issues": "No" <br>
  }'
  ### 6. Deploy with Docker:
  - Build image
docker build -t fastfood-health-api .
- Run container
docker run -p 8000:8000 fastfood-health-api
### 7. Deploy to Fly.io:
- Install flyctl <br>
curl -L https://fly.io/install.sh | sh

- Login <br>
flyctl auth login

- Launch app <br>
flyctl launch

- Deploy <br>
flyctl deploy
