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
├── README.md
└── .dockerignore
```
