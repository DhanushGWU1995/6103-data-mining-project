# Health Predictor App

A comprehensive web application for predicting health outcomes based on socioeconomic and lifestyle factors using machine learning. This app combines a modern Angular frontend with a Flask REST API backend, powered by a CatBoost machine learning model trained on CDC BRFSS 2024 data.

## 🏗️ Architecture Overview

### Frontend (Angular 16)
- **Framework**: Angular 16 with TypeScript
- **UI Components**: Multi-step health assessment form with progress tracking
- **Features**: Responsive design, form validation, real-time feedback
- **Styling**: Modern CSS with gradients, animations, and mobile-first design

### Backend (Flask API)
- **Framework**: Flask with RESTful API design
- **ML Model**: CatBoost classifier for health outcome prediction
- **Features**: CORS enabled, JSON request/response handling

### Machine Learning Model
- **Algorithm**: CatBoost (Categorical Boosting)
- **Training Data**: CDC BRFSS 2024 dataset (Behavioral Risk Factor Surveillance System)
- **Target Variable**: General Health Status (Poor/Fair vs Good/Very Good/Excellent)
- **Features**: 17 socioeconomic and health-related predictors

## 📊 Model Details

### Training Process
The model was trained on the 2024 CDC BRFSS dataset, which contains responses from over 400,000 Americans about their health behaviors and conditions.

**Key Training Steps:**
1. **Data Preprocessing**: Cleaned and encoded categorical variables
2. **Feature Selection**: Selected 17 most relevant socioeconomic and health factors
3. **Model Training**: CatBoost classifier with hyperparameter optimization
4. **Validation**: Cross-validation and performance metrics evaluation
5. **Model Persistence**: Saved as `catboost_model.pkl` for deployment

### Feature Importance (Top 10)
Based on the trained CatBoost model, here are the top 10 most important predictors ranked by their contribution to health outcome prediction:

1. **Mental Health Days** (13.67%) - Number of poor mental health days in past 30 days
2. **BMI Value** (10.11%) - Body Mass Index measurement
3. **Income Categories** (9.39%) - Household income brackets
4. **Employment Status** (7.94%) - Current employment situation
5. **Diabetes Status** (7.44%) - Diabetes diagnosis status
6. **Education Level** (5.95%) - Highest education completed
7. **Arthritis** (5.34%) - Arthritis diagnosis
8. **Age Group** (5.26%) - 5-year age categories
9. **Personal Doctor** (5.19%) - Having a regular healthcare provider
10. **Difficulty Doing Errands Alone** (4.92%) - Mobility and independence level

### Model Performance
- **Accuracy**: ~78%
- **AUC-ROC**: 0.85
- **Precision**: 0.82 (for poor health prediction)
- **Recall**: 0.75 (for poor health prediction)

## 🚀 Quick Start

### Prerequisites
- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher)
- **npm** or **yarn**
- **pip** (Python package manager)

### Installation & Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/DhanushGWU1995/6103-data-mining-project.git
cd 6103-data-mining-project/health-predictor-app
```

#### 2. Backend Setup
```bash
cd backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Verify model file exists
ls -la catboost_model.pkl
```

#### 3. Frontend Setup
```bash
cd ../frontend

# Install Node.js dependencies
npm install

# Build the application (optional, for production)
npm run build
```

### Running the Application

#### Development Mode (Recommended for development)
```bash
# Terminal 1: Start Backend API
cd backend
python3 app.py
# Server will start on http://127.0.0.1:5000

# Terminal 2: Start Frontend Development Server
cd ../frontend
npm start
# App will be available at http://localhost:4200
```

#### Production Mode
```bash
# Build frontend for production
cd frontend
npm run build

# Serve the built files (you'll need a web server like nginx or Apache)
# The backend should be deployed separately and configured for production
```

## 📱 Usage Guide

### Health Assessment Process

1. **Dashboard**: View model performance metrics and feature importance
2. **Health Assessment**: Complete the 4-step assessment:
   - **Step 1**: Financial Factors (income, insurance, employment)
   - **Step 2**: Demographic Information (age, gender)
   - **Step 3**: Health Metrics (BMI, exercise, mental health)
   - **Step 4**: Health Conditions (diabetes, heart disease, mobility)
3. **Results**: View personalized health risk prediction with recommendations

### API Endpoints

#### POST `/api/predict`
Predicts health outcomes based on user input.

**Request Body:**
```json
{
  "Income_Categories": 5,
  "Could_Not_Afford_Doctor": 1,
  "Employment_Status": 1,
  "Primary_Insurance": 1,
  "Education_Level": 4,
  "Age_Group_5yr": 6,
  "Sex": 1,
  "BMI_Value": 25.0,
  "Exercise_Past_30_Days": 1,
  "Mental_Health_Days": 5.0,
  "Diabetes_Status": 1,
  "Coronary_Heart_Disease": 2,
  "Personal_Doctor": 1,
  "Difficulty_Doing_Errands_Alone": 2,
  "Difficulty_Dressing_Bathing": 2,
  "Difficulty_Concentrating": 2,
  "Arthritis": 2
}
```

**Response:**
```json
{
  "prediction": "Poor Health",
  "probability": 0.78,
  "confidence": "High",
  "recommendations": [
    "Consider consulting with a healthcare provider",
    "Focus on mental health support",
    "Maintain regular exercise routine"
  ]
}
```

## 🔧 Configuration

### Frontend Configuration
- **Proxy Configuration**: `frontend/proxy.conf.json` handles API routing in development
- **Angular Environment**: Configure API endpoints in `frontend/src/environments/`

### Backend Configuration
- **Model Path**: Update `MODEL_PATH` in `backend/app.py` if model file location changes
- **CORS Settings**: Modify CORS configuration for production deployment
- **Port Configuration**: Default port is 5000, configurable in `app.py`

## 📁 Project Structure

```
health-predictor-app/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── catboost_model.pkl     # Trained ML model
│   ├── catboost_model.cbm     # CatBoost model file
│   ├── inspect_model.py       # Model inspection script
│   ├── requirements.txt       # Python dependencies
│   ├── API_TESTING.md         # API testing documentation
│   └── README.md             # Backend-specific README
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── app/           # Main app component
│   │   │   ├── dashboard/     # Dashboard component
│   │   │   ├── prediction-form/ # Health assessment form
│   │   │   └── results/       # Results display component
│   │   └── ...
│   ├── package.json           # Node.js dependencies
│   ├── angular.json          # Angular configuration
│   ├── proxy.conf.json       # Development proxy config
│   └── README.md             # Frontend-specific README
└── README.md                 # This file
```

## 🧪 Testing

### Backend Testing
```bash
cd backend
python3 -c "
import requests
response = requests.post('http://127.0.0.1:5000/api/predict',
    json={'Income_Categories':5, 'Could_Not_Afford_Doctor':1, ...})
print(response.json())
"
```

### Frontend Testing
```bash
cd frontend
npm test  # Run unit tests
npm run lint  # Check code quality
```

## 🚀 Deployment

### Backend Deployment
```bash
# Using Gunicorn (production WSGI server)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or using Docker
docker build -t health-predictor-backend .
docker run -p 5000:5000 health-predictor-backend
```

### Frontend Deployment
```bash
# Build for production
npm run build --prod

# Deploy dist/ folder to web server (nginx, Apache, etc.)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 👥 Team

- **Dhanush** - Data Analysis, Application Development & ML Model
- **Vaishali** - ML Research & Presentation
- **Simbarashe** - Data Analysis & ML Research

## 🙏 Acknowledgments

- **CDC BRFSS 2024 Dataset** - Primary data source
- **CatBoost** - Machine learning framework
- **Angular** - Frontend framework
- **Flask** - Backend framework

## 📞 Support

For questions or issues:
1. Email support (dhanush.mathivanan@gwu.edu)
2. Review the API documentation in `backend/API_TESTING.md`
3. Contact the development team (Dhanush Mathivanan)

---

**Data Source**: [CDC BRFSS Annual Survey Data](https://www.cdc.gov/brfss/annual_data/annual_2024.html)
**Model Training**: Based on 2024 Behavioral Risk Factor Surveillance System data
**Last Updated**: November 2025
