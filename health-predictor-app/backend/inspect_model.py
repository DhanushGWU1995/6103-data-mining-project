import joblib
import numpy as np

# Load model
model = joblib.load('catboost_model.pkl')

# Inspect
print("Model Type:", type(model))
print("Feature Names:", model.feature_names_)
print("Feature Importance:", model.get_feature_importance())

# Print feature importance in descending order
feature_importance = model.get_feature_importance()
feature_names = model.feature_names_
sorted_indices = np.argsort(feature_importance)[::-1]
print("Feature Importance (Descending):")
for i in sorted_indices:
    print(f"{feature_names[i]}: {feature_importance[i]:.2f}%")

# Test prediction
test_input = [5, 1, 1, 1, 4, 6, 1, 25.0, 1, 5.0, 1, 2, 1, 2, 2, 2, 2]
pred = model.predict([test_input])
proba = model.predict_proba([test_input])
print("Test Prediction:", pred)
print("Test Probability:", proba)