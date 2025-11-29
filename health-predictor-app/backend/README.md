# Health Predictor Backend (Flask API)

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your trained CatBoost model as `catboost_model.pkl` in this folder (or update the code to match your model filename).
4. Run the API:
   ```bash
   python app.py
   ```

## API Usage

- **POST /predict**
  - Request JSON: `{ "Income_Categories": 2, "Could_Not_Afford_Doctor": 1, ... }` (all 17 features)
  - Response JSON: `{ "prediction": "Poor Health", "probability": 0.78, "model": "CatBoost", "explanation": "..." }`

Edit `app.py` to match your model's preprocessing pipeline as needed.
