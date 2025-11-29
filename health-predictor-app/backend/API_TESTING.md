# API Testing Document for Health Prediction Model

This document provides test cases for the `/api/predict` endpoint of the health prediction backend. The endpoint accepts POST requests with JSON data containing health-related features and returns a prediction for health status.

## Endpoint Details
- **URL**: `http://127.0.0.1:5000/api/predict`
- **Method**: POST
- **Content-Type**: application/json
- **Response**: JSON object with `prediction`, `probability`, `model`, and `explanation`.

## Test Scenarios

### Scenario 1: Good Health Prediction
**Description**: This input represents a profile with favorable health factors (e.g., high income, no major health issues, regular exercise). Expected prediction: "Good Health" (probability ≤ 0.5).

**Request Body**:
```json
{
  "Income_Categories": 8,
  "Could_Not_Afford_Doctor": 2,
  "Employment_Status": 1,
  "Primary_Insurance": 1,
  "Education_Level": 6,
  "Age_Group_5yr": 5,
  "Sex": 1,
  "BMI_Value": 22.0,
  "Exercise_Past_30_Days": 1,
  "Mental_Health_Days": 0.0,
  "Diabetes_Status": 2,
  "Coronary_Heart_Disease": 2,
  "Personal_Doctor": 1,
  "Difficulty_Doing_Errands_Alone": 2,
  "Difficulty_Dressing_Bathing": 2,
  "Difficulty_Concentrating": 2,
  "Arthritis": 2
}
```

**Expected Response**:
```json
{
  "prediction": "Good Health",
  "probability": 0.3,
  "model": "CatBoost",
  "explanation": "Your predicted risk of poor health is low based on your input."
}
```

### Scenario 2: Poor Health Prediction
**Description**: This input represents a profile with adverse health factors (e.g., low income, multiple health issues, no exercise). Expected prediction: "Poor Health" (probability > 0.5).

**Request Body**:
```json
{
  "Income_Categories": 1,
  "Could_Not_Afford_Doctor": 1,
  "Employment_Status": 2,
  "Primary_Insurance": 9,
  "Education_Level": 1,
  "Age_Group_5yr": 10,
  "Sex": 2,
  "BMI_Value": 35.0,
  "Exercise_Past_30_Days": 2,
  "Mental_Health_Days": 30.0,
  "Diabetes_Status": 1,
  "Coronary_Heart_Disease": 1,
  "Personal_Doctor": 2,
  "Difficulty_Doing_Errands_Alone": 1,
  "Difficulty_Dressing_Bathing": 1,
  "Difficulty_Concentrating": 1,
  "Arthritis": 1
}
```

### Scenario 3: Moderate Health Prediction
**Description**: This input represents a profile with mixed factors (e.g., moderate income, some health issues but exercise). Expected prediction: Could be "Good Health" or "Poor Health" depending on model threshold (probability around 0.5).

**Request Body**:
```json
{
  "Income_Categories": 5,
  "Could_Not_Afford_Doctor": 1,
  "Employment_Status": 1,
  "Primary_Insurance": 1,
  "Education_Level": 4,
  "Age_Group_5yr": 6,
  "Sex": 1,
  "BMI_Value": 28.0,
  "Exercise_Past_30_Days": 1,
  "Mental_Health_Days": 10.0,
  "Diabetes_Status": 1,
  "Coronary_Heart_Disease": 2,
  "Personal_Doctor": 1,
  "Difficulty_Doing_Errands_Alone": 2,
  "Difficulty_Dressing_Bathing": 2,
  "Difficulty_Concentrating": 1,
  "Arthritis": 2
}
```

**Expected Response**:
```json
{
  "prediction": "Good Health",
  "probability": 0.45,
  "model": "CatBoost",
  "explanation": "Your predicted risk of poor health is low based on your input."
}
```

### Scenario 4: High BMI and No Exercise
**Description**: This input emphasizes obesity and lack of physical activity, which are strong risk factors. Expected prediction: "Poor Health" (probability > 0.5).

**Request Body**:
```json
{
  "Income_Categories": 3,
  "Could_Not_Afford_Doctor": 2,
  "Employment_Status": 1,
  "Primary_Insurance": 1,
  "Education_Level": 3,
  "Age_Group_5yr": 8,
  "Sex": 2,
  "BMI_Value": 40.0,
  "Exercise_Past_30_Days": 2,
  "Mental_Health_Days": 15.0,
  "Diabetes_Status": 1,
  "Coronary_Heart_Disease": 1,
  "Personal_Doctor": 1,
  "Difficulty_Doing_Errands_Alone": 1,
  "Difficulty_Dressing_Bathing": 2,
  "Difficulty_Concentrating": 2,
  "Arthritis": 1
}
```

**Expected Response**:
```json
{
  "prediction": "Poor Health",
  "probability": 0.75,
  "model": "CatBoost",
  "explanation": "Your predicted risk of poor health is high based on your input."
}
```

### Scenario 5: Young and Healthy Profile
**Description**: This input represents a young individual with optimal health factors. Expected prediction: "Good Health" (probability ≤ 0.5).

**Request Body**:
```json
{
  "Income_Categories": 7,
  "Could_Not_Afford_Doctor": 2,
  "Employment_Status": 1,
  "Primary_Insurance": 1,
  "Education_Level": 5,
  "Age_Group_5yr": 2,
  "Sex": 1,
  "BMI_Value": 23.0,
  "Exercise_Past_30_Days": 1,
  "Mental_Health_Days": 2.0,
  "Diabetes_Status": 2,
  "Coronary_Heart_Disease": 2,
  "Personal_Doctor": 1,
  "Difficulty_Doing_Errands_Alone": 2,
  "Difficulty_Dressing_Bathing": 2,
  "Difficulty_Concentrating": 2,
  "Arthritis": 2
}
```

**Expected Response**:
```json
{
  "prediction": "Good Health",
  "probability": 0.2,
  "model": "CatBoost",
  "explanation": "Your predicted risk of poor health is low based on your input."
}
```
1. Ensure the backend server is running: `python app.py`
2. Use tools like curl, Postman, or the frontend UI to send POST requests to `http://127.0.0.1:5000/api/predict` with the above JSON bodies.
3. Verify the response matches the expected prediction and probability range.

## Notes
- Probabilities are approximate and depend on the trained model.
- If the model file (`catboost_model.pkl`) is not found, a mock model is used, which always returns a probability of 0.78 (Poor Health).
