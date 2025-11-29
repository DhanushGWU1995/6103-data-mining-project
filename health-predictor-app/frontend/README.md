# Health Predictor Frontend (Angular)

## Setup

1. Make sure you have Node.js and Angular CLI installed:
   ```bash
   npm install -g @angular/cli
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the app:
   ```bash
   ng serve
   ```

## Features
- Dashboard with model performance
- Prediction form for user input (17 features)
- Results display with probability and explanation
- Visualizations (charts or static images)

## API Integration
- The app expects the Flask backend to be running at `/api/predict` (proxy or CORS setup may be needed for local dev)
