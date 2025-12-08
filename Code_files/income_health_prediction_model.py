"""
Income and Financial Factors - Health Outcome Prediction Model
================================================================
This script builds a high-accuracy statistical model to predict health outcomes
based on income and financial social determinants of health.

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("INCOME & FINANCIAL SOCIAL FACTORS - HEALTH OUTCOME PREDICTION MODEL")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================

print("\n1. LOADING DATA...")
print("-" * 60)

df = pd.read_csv('BRFSS_2024_Readable_Columns.csv')
print(f"Initial dataset shape: {df.shape}")

# ============================================================================
# STEP 2: IDENTIFY INCOME AND FINANCIAL VARIABLES
# ============================================================================

print("\n2. INCOME AND FINANCIAL SOCIAL FACTOR VARIABLES:")
print("-" * 60)

financial_variables = {
    # 'Income_Level': 'Household income level (detailed)',
    'Income_Categories': 'Grouped income categories',
    'Could_Not_Afford_Doctor': 'Could not afford to see doctor',
    'Food_Insecurity': 'Food insecurity status',
    'Own_or_Rent_Home': 'Home ownership status',
    'Employment_Status': 'Employment status',
    'Primary_Insurance': 'Primary health insurance coverage',
    'Education_Level': 'Education level (socioeconomic indicator)',
}

for var, desc in financial_variables.items():
    if var in df.columns:
        print(f"* {var}: {desc}")
        print(f"  Missing: {df[var].isnull().sum()} ({df[var].isnull().mean()*100:.2f}%)")

# ============================================================================
# STEP 3: SELECT HEALTH OUTCOME - General Health Status
# ============================================================================

print("\n3. HEALTH OUTCOME VARIABLE:")
print("-" * 60)
print("Selected: General_Health_Status")
print("Reason: This is a comprehensive health indicator that reflects overall")
print("        health and well-being, strongly associated with socioeconomic")
print("        factors. We'll predict Poor/Fair health (poor health) vs")
print("        Excellent/Very Good/Good health (good health).")

# Check distribution
if 'General_Health_Status' in df.columns:
    print(f"\nGeneral Health Status distribution:")
    print(df['General_Health_Status'].value_counts())

# ============================================================================
# STEP 4: DATA CLEANING AND PREPARATION
# ============================================================================

print("\n4. DATA CLEANING AND PREPARATION:")
print("-" * 60)

# Create working copy
df_clean = df.copy()

# Remove rows with missing General_Health_Status
df_clean = df_clean[df_clean['General_Health_Status'].notnull()]
print(f"Records after removing missing health status: {len(df_clean)}")

# Create binary health outcome: Good Health (1,2,3) vs Poor Health (4,5)
def categorize_health(val):
    if pd.isna(val):
        return np.nan
    if val in [1, 2, 3]:
        return 0  # Good health (Excellent, Very Good, Good)
    elif val in [4, 5]:
        return 1  # Poor health (Fair, Poor)
    else:
        return np.nan

df_clean['Poor_Health'] = df_clean['General_Health_Status'].apply(categorize_health)

# CRITICAL FIX: Remove rows with missing/refused codes in key variables
# According to BRFSS codebook: Different variables have different missing codes!
print("\n CRITICAL DATA CLEANING: Removing missing/refused responses")
print("   IMPORTANT: Each variable has different valid ranges!")

# Remove invalid INCOME responses 
# INCOME3 (Income_Categories): Valid values = 1-11, Missing = 77, 99
initial_count = len(df_clean)
df_clean = df_clean[~df_clean['Income_Categories'].isin([77, 99])]
print(f"   Removed {initial_count - len(df_clean)} records with invalid Income_Categories codes (77, 99)")

# Remove rows with completely missing income data
df_clean = df_clean[df_clean['Income_Categories'].notnull()]
# df_clean = df_clean[df_clean['Income_Level'].notnull()]

# Clean other categorical variables - CHECK VALID RANGES FOR EACH!
# For variables with range 1-2 or 1-3: codes 7, 9 are missing
# For variables with broader ranges: need to check codebook individually

# Could_Not_Afford_Doctor: Valid = 1,2; Missing = 7, 9
if 'Could_Not_Afford_Doctor' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Could_Not_Afford_Doctor'].isin([7, 9])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Could_Not_Afford_Doctor codes (7, 9)")

# Education_Level: Need to check valid range (likely 1-6, missing = 9)
if 'Education_Level' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Education_Level'].isin([9])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Education_Level codes (9)")

# Employment_Status: Need to check valid range (likely 1-8, missing = 9)
if 'Employment_Status' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Employment_Status'].isin([9])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Employment_Status codes (9)")

# Primary_Insurance: Need to check valid range
if 'Primary_Insurance' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Primary_Insurance'].isin([77, 99])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Primary_Insurance codes (77, 99)")

# Exercise_Past_30_Days: Valid = 1-2; Missing = 7, 9
if 'Exercise_Past_30_Days' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Exercise_Past_30_Days'].isin([7, 9])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Exercise_Past_30_Days codes (7, 9)")

# Diabetes_Status: Valid = 1-4; Missing = 7, 9
if 'Diabetes_Status' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Diabetes_Status'].isin([7, 9])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Diabetes_Status codes (7, 9)")

# Coronary_Heart_Disease: Valid = 1-2; Missing = 7, 9
if 'Coronary_Heart_Disease' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Coronary_Heart_Disease'].isin([7, 9])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Coronary_Heart_Disease codes (7, 9)")

# Personal_Doctor: Valid = 1-3; Missing = 7, 9
if 'Personal_Doctor' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Personal_Doctor'].isin([7, 9])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Personal_Doctor codes (7, 9)")

# Disability measures: Valid = 1-2; Missing = 7, 9
if 'Difficulty_Doing_Errands_Alone' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Difficulty_Doing_Errands_Alone'].isin([7, 9])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Difficulty_Doing_Errands_Alone codes (7, 9)")

if 'Difficulty_Dressing_Bathing' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Difficulty_Dressing_Bathing'].isin([7, 9])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Difficulty_Dressing_Bathing codes (7, 9)")

if 'Difficulty_Concentrating' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Difficulty_Concentrating'].isin([7, 9])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Difficulty_Concentrating codes (7, 9)")

# Chronic conditions: Valid = 1-2; Missing = 7, 9
if 'Arthritis' in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['Arthritis'].isin([7, 9])]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} records with invalid Arthritis codes (7, 9)")


# Remove rows with age > 69 (focus on working-age and early retirement population)
if 'Age_Group_5yr' in df_clean.columns:
    df_clean = df_clean[df_clean['Age_Group_5yr'] <= 10]  # Up to 65-69 age group

print(f"Records after initial cleaning: {len(df_clean)}")

# Handle BMI
if 'BMI_Category' in df_clean.columns:
    df_clean['BMI_Value'] = df_clean['BMI_Category'] / 100
    # Remove outliers
    df_clean = df_clean[(df_clean['BMI_Value'] >= 12) & (df_clean['BMI_Value'] <= 60)]

print(f"Records after BMI cleaning: {len(df_clean)}")

# Check target variable distribution
print(f"\nHealth Outcome distribution:")
print(df_clean['Poor_Health'].value_counts())
print(f"\nPoor health prevalence: {df_clean['Poor_Health'].mean()*100:.2f}%")

# ============================================================================
# STEP 5: FEATURE SELECTION
# ============================================================================

print("\n5. FEATURE SELECTION:")
print("-" * 60)

# Select features for modeling
feature_columns = [
    # Primary financial/socioeconomic variables
    'Income_Categories',
    'Could_Not_Afford_Doctor',
    # 'Food_Insecurity',                 # REMOVED: Rank #20, 1.47% - weakest feature + 55% missing data
    # 'Own_or_Rent_Home',                # REMOVED: Rank #16, 1.78% - redundant with Income_Categories
    'Employment_Status',
    'Primary_Insurance',
    'Education_Level',
    
    # Demographics and health behaviors that interact with financial factors
    'Age_Group_5yr',
    'Sex',
    'BMI_Value',
    'Exercise_Past_30_Days',
    # 'Smoked_100_Cigarettes_Lifetime',  # REMOVED: Rank #19, 1.57% - very weak predictor
    'Mental_Health_Days',
    'Diabetes_Status',
    'Coronary_Heart_Disease',
    'Personal_Doctor',
    # 'Last_Checkup',                    # REMOVED: Rank #17, 1.72% - redundant with Personal_Doctor
    
    # Disability measures (3 features) - Strong predictors of poor health
    'Difficulty_Doing_Errands_Alone',    # Rank #5 in importance (6.39%)
    'Difficulty_Dressing_Bathing',       # Rank #15 in importance (2.21%)
    'Difficulty_Concentrating',          # Rank #6 in importance (5.45%)
    # 'Blind_or_Visual_Difficulty',      # REMOVED: Not in top 20 - weakest disability measure
    
    # Chronic conditions (1 feature) - Not captured by existing diabetes/heart disease
    'Arthritis',                         # Rank #11 in importance (4.32%)
]

# Filter to available columns
available_features = [col for col in feature_columns if col in df_clean.columns]
print(f"Using {len(available_features)} features:")
for i, feat in enumerate(available_features, 1):
    print(f"  {i}. {feat}")

# Count financial variables
financial_feats = [f for f in available_features if f in financial_variables.keys()]
print(f"\nFinancial/socioeconomic features: {len(financial_feats)}")

# ============================================================================
# STEP 6: PREPARE DATA FOR MODELING
# ============================================================================

print("\n6. PREPARING DATA FOR MODELING:")
print("-" * 60)

# Remove rows with missing target
df_clean = df_clean.dropna(subset=['Poor_Health'])

# Create feature matrix and target
X = df_clean[available_features].copy()
y = df_clean['Poor_Health'].copy()

print(f"Initial feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# Encode categorical variables
label_encoders = {}
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

print(f"\nEncoding {len(categorical_cols)} categorical variables...")
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Get indices of categorical columns for CatBoost
cat_features_idx = [X.columns.get_loc(col) for col in categorical_cols]
# Handle missing values in features
print("Handling missing values...")
for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype in ['int64', 'float64']:
            X[col].fillna(X[col].median(), inplace=True)
            print(f"  Filled missing numeric '{col}' with median.")
        else:
            X[col].fillna(X[col].mode()[0], inplace=True)
            print(f"  Filled missing categorical '{col}' with mode.")

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"\nClass distribution (BEFORE balancing):")
print(f"  Good Health: {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")
print(f"  Poor Health: {(y==1).sum()} ({(y==1).mean()*100:.2f}%)")

# ============================================================================
# STEP 6.5: HANDLE CLASS IMBALANCE - STATE-BASED UNDERSAMPLING
# ============================================================================

print("\n6.5. HANDLING CLASS IMBALANCE (State-Based Undersampling):")
print("-" * 60)

# Check if State_Code column exists in df_clean
if 'State_Code' in df_clean.columns:
    print("State_Code column found - performing state-based undersampling...")
    
    # Get indices for poor health and good health
    poor_health_indices = df_clean[df_clean['Poor_Health'] == 1].index
    good_health_indices = df_clean[df_clean['Poor_Health'] == 0].index
    
    print(f"Original - Poor Health: {len(poor_health_indices)}, Good Health: {len(good_health_indices)}")
    
    # Keep all poor health records
    balanced_indices = list(poor_health_indices)
    
    # For each poor health record, randomly select one good health record from the same state
    # WITHOUT REPLACEMENT (no duplicates)
    np.random.seed(42)  # For reproducibility
    sampled_good_health = []
    already_sampled = set()  # Track already sampled indices to avoid duplicates
    
    # Group poor health records by state for efficient sampling
    state_groups = df_clean.loc[poor_health_indices].groupby('State_Code').size().to_dict()
    print(f"\nPoor health records by state: {len(state_groups)} states")
    
    # For each state, create a pool of available good health records
    state_pools = {}
    for state in state_groups.keys():
        pool = df_clean[
            (df_clean['Poor_Health'] == 0) & 
            (df_clean['State_Code'] == state)
        ].index.tolist()
        state_pools[state] = pool
        print(f"  State {state}: {state_groups[state]} poor health, {len(pool)} good health available")
    
    # Sample without replacement for each poor health record
    fallback_count = 0
    duplicate_prevention_count = 0
    
    for idx in poor_health_indices:
        state = df_clean.loc[idx, 'State_Code']
        
        # Get available good health records from same state (not yet sampled)
        available_in_state = [i for i in state_pools.get(state, []) if i not in already_sampled]
        
        if len(available_in_state) > 0:
            # Randomly select one from available pool in same state
            selected_idx = np.random.choice(available_in_state)
            sampled_good_health.append(selected_idx)
            already_sampled.add(selected_idx)
        else:
            # If no more good health records available in same state, try any other state
            available_any_state = [i for i in good_health_indices if i not in already_sampled]
            
            if len(available_any_state) > 0:
                selected_idx = np.random.choice(available_any_state)
                sampled_good_health.append(selected_idx)
                already_sampled.add(selected_idx)
                fallback_count += 1
            else:
                # Edge case: if all good health records are exhausted (shouldn't happen)
                print(f"  WARNING: All good health records exhausted at poor health record {idx}")
                break
    
    print(f"\nSampling complete:")
    print(f"  Total good health records sampled: {len(sampled_good_health)}")
    print(f"  Unique records: {len(set(sampled_good_health))}")
    print(f"  Records sampled from different state (fallback): {fallback_count}")
    
    if len(sampled_good_health) != len(set(sampled_good_health)):
        print(f" WARNING: {len(sampled_good_health) - len(set(sampled_good_health))} duplicate records found!")
    else:
        print(f" No duplicates - all sampled records are unique!")
    
    # Combine poor health and sampled good health indices
    balanced_indices.extend(sampled_good_health)
    
    # Create balanced dataset
    df_balanced = df_clean.loc[balanced_indices].copy()
    
    print(f"After state-based undersampling:")
    print(f"  Total records: {len(df_balanced)}")
    print(f"  Poor Health: {(df_balanced['Poor_Health'] == 1).sum()}")
    print(f"  Good Health: {(df_balanced['Poor_Health'] == 0).sum()}")
    print(f"  Balance ratio: {(df_balanced['Poor_Health'] == 1).sum() / len(df_balanced) * 100:.2f}%")
    
    # Update X and y with balanced data
    X = df_balanced[available_features].copy()
    y = df_balanced['Poor_Health'].copy()
    
    # Re-encode if needed (after balancing)
    for col in categorical_cols:
        if col in X.columns:
            le = label_encoders[col]
            X[col] = X[col].astype(str)
            X[col] = le.transform(X[col])
    
    # Re-handle missing values
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in ['int64', 'float64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0], inplace=True)
    
    print(f"\nBalanced feature matrix shape: {X.shape}")
    print(f"Balanced target variable shape: {y.shape}")
    
else:
    print("   State_Code column NOT found - skipping state-based undersampling")
    print("   To enable this feature:")
    print("   1. Update xpt_to_csv_data_sample.py to include '_STATE' variable")
    print("   2. Add 'State_Code' to column_mapping")
    print("   3. Regenerate BRFSS_2024_Readable_Columns.csv")
    print("   Proceeding with original imbalanced dataset...")

print(f"\nFinal Class distribution (AFTER balancing):")
print(f"  Good Health: {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")
print(f"  Poor Health: {(y==1).sum()} ({(y==1).mean()*100:.2f}%)")

# ============================================================================
# STEP 7: TRAIN-TEST SPLIT
# ============================================================================

print("\n7. SPLITTING DATA:")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train poor health rate: {y_train.mean()*100:.2f}%")
print(f"Test poor health rate: {y_test.mean()*100:.2f}%")

# ============================================================================
# STEP 8: BUILD AND EVALUATE MODELS
# ============================================================================

print("\n8. MODEL BUILDING AND EVALUATION:")
print("=" * 80)

model_results = {}

# -------------------------
# Model 1: Logistic Regression
# -------------------------
print("\n[Model 1] Logistic Regression")
print("-" * 60)

# Train Logistic Regression (No class weights needed - dataset is balanced 50/50)
lr_model = LogisticRegression(
    max_iter=2000, 
    random_state=42, 
    C=1.0,
    solver='lbfgs'
)
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Calculate metrics
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_pred_proba_lr)

print(f"Accuracy: {lr_accuracy*100:.2f}%")
print(f"AUC-ROC: {lr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Good Health', 'Poor Health']))

# Show confusion matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nConfusion Matrix:")
print(f"                 Predicted Good  Predicted Poor")
print(f"Actual Good      {cm_lr[0,0]:14d}  {cm_lr[0,1]:14d}")
print(f"Actual Poor      {cm_lr[1,0]:14d}  {cm_lr[1,1]:14d}")

model_results['Logistic Regression'] = {
    'accuracy': lr_accuracy,
    'auc': lr_auc,
    'model': lr_model,
    'predictions': y_pred_lr
}

# -------------------------
# Model 2: Random Forest (Enhanced)
# -------------------------
print("\n[Model 2] Random Forest Classifier (OPTIMIZED FOR POOR HEALTH)")
print("-" * 60)

# Random Forest (No class weights needed - dataset is balanced 50/50)
rf_model = RandomForestClassifier(
    n_estimators=300,  # Increased from 200
    max_depth=25,      # Increased from 20
    min_samples_split=8,  # Decreased from 10 for more splits
    min_samples_leaf=3,   # Decreased from 5
    random_state=42,
    n_jobs=-1,
    max_features='sqrt',
    bootstrap=True
)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Calculate metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_pred_proba_rf)

print(f"Accuracy: {rf_accuracy*100:.2f}%")
print(f"AUC-ROC: {rf_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Good Health', 'Poor Health']))

model_results['Random Forest'] = {
    'accuracy': rf_accuracy,
    'auc': rf_auc,
    'model': rf_model,
    'predictions': y_pred_rf
}

# Feature importance
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(20).to_string(index=False))

print("\n The confusion matrix for Random Forest Classifier is as follows:")    
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:")
print(f"                 Predicted Good  Predicted Poor")
print(f"Actual Good      {cm_rf[0,0]:14d}  {cm_rf[0,1]:14d}")
print(f"Actual Poor      {cm_rf[1,0]:14d}  {cm_rf[1,1]:14d}")


# ========================================================================
# Model 3: CatBoost Classifier
# ========================================================================
print("\n[Model 3] CatBoost Classifier")
print("-" * 60)

# Create CatBoost Pool objects
train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_features_idx)

# CatBoost (NO class weights needed - dataset is balanced 50/50)
cb_model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

cb_model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# Predictions
y_pred_proba_cb = cb_model.predict_proba(test_pool)[:, 1]
y_pred_cb = (y_pred_proba_cb > 0.5).astype(int)

# Metrics
cb_accuracy = accuracy_score(y_test, y_pred_cb)
cb_auc = roc_auc_score(y_test, y_pred_proba_cb)

print(f"Accuracy: {cb_accuracy*100:.2f}%")
print(f"AUC-ROC: {cb_auc:.4f}")
print("\nClassification Report (CatBoost):")
print(classification_report(y_test, y_pred_cb, target_names=['Good Health', 'Poor Health']))

# Confusion Matrix for CatBoost
cm_cb = confusion_matrix(y_test, y_pred_cb)
print("\nConfusion Matrix (CatBoost):")
print(f"                 Predicted Good  Predicted Poor")
print(f"Actual Good      {cm_cb[0,0]:14d}  {cm_cb[0,1]:14d}")
print(f"Actual Poor      {cm_cb[1,0]:14d}  {cm_cb[1,1]:14d}")

model_results['CatBoost'] = {
    'accuracy': cb_accuracy,
    'auc': cb_auc,
    'model': cb_model,
    'predictions': y_pred_cb
}

# ===================== MODEL EXPORT FOR WEB APP =============================
import os
catboost_export_dir = os.path.join(os.path.dirname(__file__), 'health-predictor-app', 'backend')
cbm_path = os.path.join(catboost_export_dir, 'catboost_model.cbm')
pkl_path = os.path.join(catboost_export_dir, 'catboost_model.pkl')

print(f"\nExporting CatBoost model to: {cbm_path} and {pkl_path}")
cb_model.save_model(cbm_path)
import joblib
joblib.dump(cb_model, pkl_path)
print("Model export complete. You can now use these files in the Flask backend.")

# ============================================================================