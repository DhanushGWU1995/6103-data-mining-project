"""
Visualization Script for Income-Health Relationship
This creates charts showing the relationship between financial factors and health outcomes
"""

from matplotlib.patches import Patch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("Creating visualizations for income-health relationship...")

# Load data
df = pd.read_csv('BRFSS_2024_Readable_Columns.csv')

# Clean data - MATCHING income_health_prediction_model.py cleaning steps
df_clean = df.copy()

# Remove rows with missing General_Health_Status
df_clean = df_clean[df_clean['General_Health_Status'].notnull()]

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


df_clean['Poor_Health'] = df_clean['General_Health_Status'].apply(
    categorize_health)

# CRITICAL DATA CLEANING: Remove invalid/missing codes (matching model file)
# Remove invalid INCOME responses (77, 99 are missing codes)
df_clean = df_clean[~df_clean['Income_Categories'].isin([77, 99])]
df_clean = df_clean[df_clean['Income_Categories'].notnull()]

# Remove invalid codes from other variables
if 'Could_Not_Afford_Doctor' in df_clean.columns:
    df_clean = df_clean[~df_clean['Could_Not_Afford_Doctor'].isin([7, 9])]

if 'Education_Level' in df_clean.columns:
    df_clean = df_clean[~df_clean['Education_Level'].isin([9])]

if 'Employment_Status' in df_clean.columns:
    df_clean = df_clean[~df_clean['Employment_Status'].isin([9])]

if 'Primary_Insurance' in df_clean.columns:
    df_clean = df_clean[~df_clean['Primary_Insurance'].isin([77, 99])]

if 'Exercise_Past_30_Days' in df_clean.columns:
    df_clean = df_clean[~df_clean['Exercise_Past_30_Days'].isin([7, 9])]

if 'Diabetes_Status' in df_clean.columns:
    df_clean = df_clean[~df_clean['Diabetes_Status'].isin([7, 9])]

if 'Coronary_Heart_Disease' in df_clean.columns:
    df_clean = df_clean[~df_clean['Coronary_Heart_Disease'].isin([7, 9])]

if 'Personal_Doctor' in df_clean.columns:
    df_clean = df_clean[~df_clean['Personal_Doctor'].isin([7, 9])]

if 'Difficulty_Doing_Errands_Alone' in df_clean.columns:
    df_clean = df_clean[~df_clean['Difficulty_Doing_Errands_Alone'].isin([
                                                                         7, 9])]

if 'Difficulty_Dressing_Bathing' in df_clean.columns:
    df_clean = df_clean[~df_clean['Difficulty_Dressing_Bathing'].isin([7, 9])]

if 'Difficulty_Concentrating' in df_clean.columns:
    df_clean = df_clean[~df_clean['Difficulty_Concentrating'].isin([7, 9])]

if 'Arthritis' in df_clean.columns:
    df_clean = df_clean[~df_clean['Arthritis'].isin([7, 9])]

# Remove rows with age > 69 (focus on working-age and early retirement)
if 'Age_Group_5yr' in df_clean.columns:
    df_clean = df_clean[df_clean['Age_Group_5yr'] <= 10]

# Handle BMI outliers
if 'BMI_Category' in df_clean.columns:
    df_clean['BMI_Value'] = df_clean['BMI_Category'] / 100
    df_clean = df_clean[(df_clean['BMI_Value'] >= 12) &
                        (df_clean['BMI_Value'] <= 60)]

# Remove rows with missing Poor_Health
df_clean = df_clean.dropna(subset=['Poor_Health'])

print(f"Data cleaning complete (Step 1-6): {len(df_clean)} records")
print(
    f"Poor health prevalence BEFORE balancing: {df_clean['Poor_Health'].mean()*100:.2f}%")
print(
    f"  Good Health: {(df_clean['Poor_Health'] == 0).sum()} ({(df_clean['Poor_Health'] == 0).mean()*100:.2f}%)")
print(
    f"  Poor Health: {(df_clean['Poor_Health'] == 1).sum()} ({(df_clean['Poor_Health'] == 1).mean()*100:.2f}%)")

# ============================================================================
# STEP 6.5: STATE-BASED UNDERSAMPLING (Matching model file)
# ============================================================================
print("\nStep 6.5: Applying State-Based Undersampling (matching model file)...")

# Check if State_Code column exists
if 'State_Code' in df_clean.columns:
    print("State_Code column found - performing state-based undersampling...")

    # Get indices for poor health and good health
    poor_health_indices = df_clean[df_clean['Poor_Health'] == 1].index
    good_health_indices = df_clean[df_clean['Poor_Health'] == 0].index

    print(
        f"Original - Poor Health: {len(poor_health_indices)}, Good Health: {len(good_health_indices)}")

    # Keep all poor health records
    balanced_indices = list(poor_health_indices)

    # For each poor health record, randomly select one good health record from the same state
    # WITHOUT REPLACEMENT (no duplicates)
    np.random.seed(42)  # For reproducibility (same as model)
    sampled_good_health = []
    already_sampled = set()

    # Group poor health records by state
    state_groups = df_clean.loc[poor_health_indices].groupby(
        'State_Code').size().to_dict()

    # Create pools of available good health records per state
    state_pools = {}
    for state in state_groups.keys():
        pool = df_clean[
            (df_clean['Poor_Health'] == 0) &
            (df_clean['State_Code'] == state)
        ].index.tolist()
        state_pools[state] = pool

    # Sample without replacement for each poor health record
    fallback_count = 0

    for idx in poor_health_indices:
        state = df_clean.loc[idx, 'State_Code']

        # Get available good health records from same state (not yet sampled)
        available_in_state = [i for i in state_pools.get(
            state, []) if i not in already_sampled]

        if len(available_in_state) > 0:
            # Randomly select one from available pool in same state
            selected_idx = np.random.choice(available_in_state)
            sampled_good_health.append(selected_idx)
            already_sampled.add(selected_idx)
        else:
            # Fallback to any other state if needed
            available_any_state = [
                i for i in good_health_indices if i not in already_sampled]
            if len(available_any_state) > 0:
                selected_idx = np.random.choice(available_any_state)
                sampled_good_health.append(selected_idx)
                already_sampled.add(selected_idx)
                fallback_count += 1

    # Combine poor health and sampled good health indices
    balanced_indices.extend(sampled_good_health)

    # Create balanced dataset
    df_balanced = df_clean.loc[balanced_indices].copy()

    print(f"After state-based undersampling:")
    print(f"  Total records: {len(df_balanced)}")
    print(f"  Poor Health: {(df_balanced['Poor_Health'] == 1).sum()}")
    print(f"  Good Health: {(df_balanced['Poor_Health'] == 0).sum()}")
    print(
        f"  Balance ratio: {(df_balanced['Poor_Health'] == 1).sum() / len(df_balanced) * 100:.2f}%")
    print(f"  Unique good health records: {len(set(sampled_good_health))}")
    print(f"  Fallback samplings: {fallback_count}")

    # Use balanced dataset for visualizations
    df_clean = df_balanced.copy()

else:
    print("WARNING: State_Code column NOT found - skipping state-based undersampling")
    print("Visualizations will use UNBALANCED data (different from model)")

print(f"\nFinal dataset for visualizations: {len(df_clean)} records")
print(
    f"Final poor health prevalence: {df_clean['Poor_Health'].mean()*100:.2f}%")
print("="*60)

# Income categories mapping (based on output)
income_mapping = {
    1.0: 'Less than $15K',
    2.0: '$10K-$15K',
    3.0: '$15K-$20K',
    4.0: '$20K-$25K',
    5.0: '$25K-$35K',
    6.0: '$35K-$50K',
    7.0: '$50K-$75K',
    8.0: '$75K-$100K',
    9.0: '$100K-$150K',
    10.0: '$150K-$200K',
    11.0: '$200K+',
    77.0: 'Unknown',
    99.0: 'Refused'
}

df_clean['Income_Label'] = df_clean['Income_Categories'].map(income_mapping)

# Calculate poor health rates by income
income_health = df_clean.groupby('Income_Categories')['Poor_Health'].agg([
    ('Count', 'count'),
    ('Poor_Health_Rate', lambda x: x.mean() * 100)
]).reset_index()

income_health['Income_Label'] = income_health['Income_Categories'].map(
    income_mapping)

# Sort by income level (excluding unknown/refused)
income_health_sorted = income_health[~income_health['Income_Categories'].isin(
    [77.0, 99.0])].sort_values('Income_Categories')

# FIGURE 1: Poor Health Rate by Income Category

plt.figure(figsize=(14, 7))
# Color: red >60%, yellow 40-60%, green <40%
bar_colors = ['#d32f2f' if rate > 60 else '#ffe600' if rate > 40 else '#4caf50'
              for rate in income_health_sorted['Poor_Health_Rate']]
bars = plt.bar(range(len(income_health_sorted)),
               income_health_sorted['Poor_Health_Rate'],
               color=bar_colors,
               edgecolor='black',
               linewidth=1.2)

plt.xlabel('Income Category', fontsize=14, fontweight='bold')
plt.ylabel('Poor Health Rate (%)', fontsize=14, fontweight='bold')
plt.title('Poor Health Prevalence by Income Level\nClear Gradient: Lower Income = Higher Poor Health Rate',
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(income_health_sorted)),
           income_health_sorted['Income_Label'],
           rotation=45, ha='right')
# Set y-axis to fit up to 90% (with some headroom)
plt.ylim(0, 90)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add color legend
legend_elements = [
    Patch(facecolor='#d32f2f', edgecolor='black', label='High (>60%)'),
    Patch(facecolor='#ffe600', edgecolor='black', label='Moderate (40-60%)'),
    Patch(facecolor='#4caf50', edgecolor='black', label='Low (<40%)')
]
plt.legend(handles=legend_elements, loc='upper right',
           title='Poor Health Rate')

# Add trend line
x_pos = range(len(income_health_sorted))
y_vals = income_health_sorted['Poor_Health_Rate'].values
z = np.polyfit(x_pos, y_vals, 2)
p = np.poly1d(z)
plt.plot(x_pos, p(x_pos), "r--", alpha=0.8, linewidth=2, label='Trend')
plt.legend()

plt.tight_layout()
plt.savefig('income_health_gradient.png', dpi=300, bbox_inches='tight')
print("Saved: income_health_gradient.png")

# FIGURE 2: Healthcare Affordability Impact
if 'Could_Not_Afford_Doctor' in df_clean.columns:
    plt.figure(figsize=(10, 6))

    afford_health = df_clean[df_clean['Could_Not_Afford_Doctor'].isin([1.0, 2.0])].groupby('Could_Not_Afford_Doctor')['Poor_Health'].agg([
        ('Poor_Health_Rate', lambda x: x.mean() * 100)
    ]).reset_index()

    afford_labels = {1.0: 'Could NOT\nAfford Doctor',
                     2.0: 'Could\nAfford Doctor'}
    afford_health['Label'] = afford_health['Could_Not_Afford_Doctor'].map(
        afford_labels)

    bars = plt.bar(afford_health['Label'],
                   afford_health['Poor_Health_Rate'],
                   color=['#d32f2f', '#4caf50'],
                   edgecolor='black',
                   linewidth=2,
                   width=0.5)

    plt.ylabel('Poor Health Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Impact of Healthcare Affordability on Health Outcomes\nFinancial Barriers Double Poor Health Risk',
              fontsize=16, fontweight='bold', pad=20)
    # Dynamically set y-axis limit to 10% above the max value, minimum 50, max 100
    max_val = afford_health['Poor_Health_Rate'].max()
    ylim_top = max(50, min(100, max_val * 1.10))
    plt.ylim(0, ylim_top)
    plt.grid(axis='y', alpha=0.3)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontweight='bold', fontsize=14)

    # Add difference annotation
    if len(afford_health) == 2:
        diff = afford_health['Poor_Health_Rate'].iloc[0] - \
            afford_health['Poor_Health_Rate'].iloc[1]
        plt.text(0.5, max(afford_health['Poor_Health_Rate']) * 0.9,
                 f'Difference: {diff:.1f} percentage points',
                 ha='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('healthcare_affordability_impact.png',
                dpi=300, bbox_inches='tight')
    print("Saved: healthcare_affordability_impact.png")

# FIGURE 3: Model Performance Comparison (Updated with 3 models - Balanced Dataset)
plt.figure(figsize=(12, 7))

models = ['Logistic\nRegression', 'Random\nForest', 'CatBoost']
accuracies = [75.96, 77.49, 78.11]
aucs = [0.8418, 0.8549, 0.8613]

x = np.arange(len(models))
width = 0.35

bars1 = plt.bar(x - width/2, accuracies, width, label='Accuracy (%)',
                color='#2196f3', edgecolor='black', linewidth=1.5)
bars2 = plt.bar(x + width/2, [a*100 for a in aucs], width, label='AUC-ROC (×100)',
                color='#ff9800', edgecolor='black', linewidth=1.5)

plt.ylabel('Score (%)', fontsize=14, fontweight='bold')
plt.title('Model Performance Comparison - Balanced Dataset (50/50 Split)\nCatBoost Achieves Best Performance: 78.11% Accuracy, 0.8613 AUC-ROC',
          fontsize=15, fontweight='bold', pad=20)
plt.xticks(x, models)
plt.ylim(0, 100)
plt.axhline(y=75, color='g', linestyle='--',
            linewidth=2, label='75% Baseline', alpha=0.7)
plt.legend(loc='lower right', fontsize=11)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add annotation about dataset balancing
plt.text(1, 10, 'Dataset: 83,832 records\n(State-based balanced: 50% poor health, 50% good health)',
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_performance_comparison.png")

# FIGURE 4: Feature Importance (Based on Random Forest Model - Updated Results)
plt.figure(figsize=(12, 8))

# Actual feature importance from your Random Forest model (Latest Run)
features = ['BMI_Value', 'Income_Categories', 'Employment_Status', 'Mental_Health_Days',
            'Difficulty_Concentrating', 'Age_Group_5yr', 'Difficulty_Doing_Errands_Alone',
            'Primary_Insurance', 'Diabetes_Status', 'Arthritis',
            'Education_Level', 'Exercise_Past_30_Days', 'Personal_Doctor',
            'Could_Not_Afford_Doctor', 'Sex', 'Difficulty_Dressing_Bathing',
            'Coronary_Heart_Disease']
importances = [14.69, 13.02, 10.66, 8.68, 6.50, 5.94, 5.75, 5.46, 5.45, 5.25,
               4.45, 4.29, 2.69, 2.66, 1.76, 1.54, 1.22]

# Mark which are income/financial/socioeconomic factors
financial = [False, True, True, False, False, False, False, True, False, False,
             True, False, False, True, False, False, False]

# Display top 10 only
features_top10 = features[:10]
importances_top10 = importances[:10]
financial_top10 = financial[:10]

colors = ['#ff5722' if f else '#2196f3' for f in financial_top10]

bars = plt.barh(features_top10, importances_top10,
                color=colors, edgecolor='black', linewidth=1.2)

plt.xlabel('Feature Importance (%)', fontsize=14, fontweight='bold')
plt.title('Top 10 Predictive Features - Random Forest Model (77.49% Accuracy)\nRed = Income/Financial/Socioeconomic Factors',
          fontsize=15, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
             f'{width:.2f}%',
             ha='left', va='center', fontweight='bold', fontsize=10)

# Calculate combined importance of financial factors in top 10
financial_importance_top10 = sum(
    [imp for imp, fin in zip(importances_top10, financial_top10) if fin])

# Add legend
legend_elements = [
    Patch(facecolor='#ff5722', edgecolor='black',
          label=f'Income/Financial Factors ({financial_importance_top10:.2f}% combined)'),
    Patch(facecolor='#2196f3', edgecolor='black',
          label='Health/Demographic Factors')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance.png")

# FIGURE 5: Income-Related Factors Impact on Health (Updated with actual importance values)
plt.figure(figsize=(12, 7))

# Financial/socioeconomic features and their importance (from latest Random Forest run)
financial_features = ['Income\nCategories', 'Employment\nStatus', 'Primary\nInsurance',
                      'Education\nLevel', 'Could Not\nAfford Doctor']
financial_importance = [13.02, 10.66, 5.46,
                        4.45, 2.66]  # Based on your latest model

bars = plt.bar(range(len(financial_features)), financial_importance,
               color=['#d32f2f', '#e64a19', '#f57c00', '#ff9800', '#ffa726'],
               edgecolor='black', linewidth=1.5)

plt.xlabel('Income & Financial Factors', fontsize=14, fontweight='bold')
plt.ylabel('Feature Importance (%)', fontsize=14, fontweight='bold')
plt.title('Income & Financial Factors as Health Predictors\nCombined Importance: 36.25% of Model Predictions',
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(financial_features)), financial_features)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add annotation
combined_importance = sum(financial_importance)
plt.text(len(financial_features)/2, max(financial_importance) * 0.85,
         f'Income & financial factors account for\n{combined_importance:.2f}% of health prediction accuracy\n(More than 1/3 of total)',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('financial_factors_importance.png', dpi=300, bbox_inches='tight')
print("Saved: financial_factors_importance.png")

# FIGURE 6: Employment Status Impact on Health
if 'Employment_Status' in df_clean.columns:
    plt.figure(figsize=(14, 7))

    # Clean employment data (remove missing codes)
    employment_clean = df_clean[~df_clean['Employment_Status'].isin([9.0])]

    employment_mapping = {
        1.0: 'Employed\nfor wages',
        2.0: 'Self-\nemployed',
        3.0: 'Out of work\n1yr+',
        4.0: 'Out of work\n<1yr',
        5.0: 'Homemaker',
        6.0: 'Student',
        7.0: 'Retired',
        8.0: 'Unable to\nwork'
    }

    employment_clean['Employment_Label'] = employment_clean['Employment_Status'].map(
        employment_mapping)

    employment_health = employment_clean.groupby('Employment_Status')['Poor_Health'].agg([
        ('Count', 'count'),
        ('Poor_Health_Rate', lambda x: x.mean() * 100)
    ]).reset_index()

    employment_health['Employment_Label'] = employment_health['Employment_Status'].map(
        employment_mapping)
    employment_health = employment_health.sort_values('Poor_Health_Rate')

    # Color code based on health rate (yellow <50%, orange 50-70%, red >70%)
    colors = ['#ffe600' if rate < 50 else '#ff9800' if rate < 70 else '#d32f2f'
              for rate in employment_health['Poor_Health_Rate']]

    bars = plt.bar(range(len(employment_health)),
                   employment_health['Poor_Health_Rate'],
                   color=colors,
                   edgecolor='black',
                   linewidth=1.2)

    plt.xlabel('Employment Status', fontsize=14, fontweight='bold')
    plt.ylabel('Poor Health Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Employment Status Impact on Health Outcomes\nUnemployment and Inability to Work Show Highest Poor Health Rates',
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(range(len(employment_health)),
               employment_health['Employment_Label'])
    plt.ylim(0, max(employment_health['Poor_Health_Rate']) * 1.15)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ffe600', edgecolor='black', label='Low (<50%)'),
        Patch(facecolor='#ff9800', edgecolor='black', label='Moderate (50-70%)'),
        Patch(facecolor='#d32f2f', edgecolor='black', label='High (>70%)')
    ]
    plt.legend(handles=legend_elements, loc='upper left',
               title='Health Risk Level')

    plt.tight_layout()
    plt.savefig('employment_health_impact.png', dpi=300, bbox_inches='tight')
    print("Saved: employment_health_impact.png")

# ============================================================================

print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)
print("\nGenerated 6 visualization files:")
print("  1. income_health_gradient.png - Income vs health relationship")
print("  2. healthcare_affordability_impact.png - Affordability impact")
print("  3. model_performance_comparison.png - Model accuracy comparison (3 models)")
print("  4. feature_importance.png - Top predictive features")
print("  5. financial_factors_importance.png - Financial factors breakdown")
print("  6. employment_health_impact.png - Employment status impact")
print("\n All visualizations saved successfully!")
