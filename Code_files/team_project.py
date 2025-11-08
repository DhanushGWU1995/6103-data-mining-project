import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


# Import data

df = pd.read_csv('BRFSS_2024_Readable_Columns.csv')

# Education Level (EDUCA): 1=Never attended, 2=Grades 1-8, 3=Grades 9-11, 4=Grade 12/GED, 5=Some college, 6=College grad, 7=Don't know/Refused
if 'Education_Level' in df.columns:
	edu_map = {
		1: 'Never attended',
		2: 'Grades 1-8',
		3: 'Grades 9-11',
		4: 'Grade 12/GED',
		5: 'Some college',
		6: 'College graduate',
		7: 'Don\'t know/Refused',
		9: 'Missing'
	}
	df['Education_Level'] = df['Education_Level'].map(edu_map)

# Employment Status (EMPLOY1): 1=Employed, 2=Self-employed, 3=Out of work, 4=Out of work >1yr, 5=Homemaker, 6=Student, 7=Retired, 8=Unable to work, 9=Refused
if 'Employment_Status' in df.columns:
	emp_map = {
		1: 'Employed', 2: 'Self-employed', 3: 'Out of work (>1yr)', 4: 'Out of work (<1yr)',
		5: 'Homemaker', 6: 'Student', 7: 'Retired', 8: 'Unable to work', 9: 'Refused', 77: 'Don\'t know', 99: 'Missing'
	}
	df['Employment_Status'] = df['Employment_Status'].map(emp_map)

# Marital Status (MARITAL): 1=Married, 2=Divorced, 3=Widowed, 4=Separated, 5=Never married, 6=Unmarried couple, 9=Refused
if 'Marital_Status' in df.columns:
	mar_map = {
		1: 'Married', 2: 'Divorced', 3: 'Widowed', 4: 'Separated', 5: 'Never married', 6: 'Unmarried couple', 9: 'Refused', 77: 'Don\'t know', 99: 'Missing'
	}
	df['Marital_Status'] = df['Marital_Status'].map(mar_map)

# Veteran Status (VETERAN3): 1=Yes, 2=No, 7=Don't know, 9=Refused
if 'Veteran_Status' in df.columns:
	vet_map = {1: 'Yes', 2: 'No', 7: 'Don\'t know', 9: 'Refused'}
	df['Veteran_Status'] = df['Veteran_Status'].map(vet_map)

# Own or Rent Home (RENTHOM1): 1=Own, 2=Rent, 3=Other, 7=Don't know, 9=Refused
if 'Own_or_Rent_Home' in df.columns:
	rent_map = {1: 'Own', 2: 'Rent', 3: 'Other', 7: 'Don\'t know', 9: 'Refused'}
	df['Own_or_Rent_Home'] = df['Own_or_Rent_Home'].map(rent_map)

# Primary Health Insurance (PRIMINS2): 1=Yes, 2=No, 7=Don't know, 9=Refused
if 'Primary_Insurance' in df.columns:
	insurance_map = {
		1: 'Employer/Union Plan',
		2: 'Private Plan',
		3: 'Medicare',
		4: 'Medigap',
		5: 'Medicaid',
		6: 'CHIP',
		7: 'Military/VA',
		8: 'Indian Health Service',
		9: 'State Plan',
		10: 'Other Government',
		88: 'No Coverage',
		77: "Don't know/Not Sure",
		99: 'Refused'
	}
	df['Primary_Insurance'] = df['Primary_Insurance'].map(insurance_map)

# Personal Doctor (PERSDOC3): 1=Yes, only one, 2=More than one, 3=None, 7=Don't know, 9=Refused
if 'Personal_Doctor' in df.columns:
	doc_map = {1: 'Yes, only one', 2: 'More than one',
            3: 'None', 7: 'Don\'t know', 9: 'Refused'}
	df['Personal_Doctor'] = df['Personal_Doctor'].map(doc_map)

# Could Not Afford Doctor (MEDCOST1): 1=Yes, 2=No, 7=Don't know, 9=Refused
if 'Could_Not_Afford_Doctor' in df.columns:
	medcost_map = {1: 'Yes', 2: 'No', 7: 'Don\'t know', 9: 'Refused'}
	df['Could_Not_Afford_Doctor'] = df['Could_Not_Afford_Doctor'].map(medcost_map)

# Last Checkup (CHECKUP1): 1=Within past year, 2=Within past 2 years, 3=Within past 5 years, 4=5 or more years ago, 7=Don't know, 9=Refused
if 'Last_Checkup' in df.columns:
	checkup_map = {1: 'Within past year', 2: 'Within past 2 years',
                3: 'Within past 5 years', 4: '5+ years ago', 7: 'Don\'t know', 9: 'Refused'}
	df['Last_Checkup'] = df['Last_Checkup'].map(checkup_map)

# ACE: Household member depressed (ACEDEPRS): 1=Yes, 2=No, 7=Don't know, 9=Refused
if 'ACE_Depressed_Household' in df.columns:
	acedep_map = {1: 'Yes', 2: 'No', 7: 'Don\'t know', 9: 'Refused'}
	df['ACE_Depressed_Household'] = df['ACE_Depressed_Household'].map(acedep_map)

# Initialize categorical variables as in R code (after loading df)
df['Sex'] = pd.Categorical(df['Sex'], categories=[1, 2], ordered=True)
df['Sex'] = df['Sex'].cat.rename_categories(['Male', 'Female'])

df['Diabetes_Status'] = pd.Categorical(
	df['Diabetes_Status'], categories=[1, 2, 3, 4], ordered=True)
df['Diabetes_Status'] = df['Diabetes_Status'].cat.rename_categories(
	['Yes', 'Yes/borderline', 'Yes/pregnancy only', 'No'])

df['Smoking_Status'] = pd.Categorical(
	df['Smoked_100_Cigarettes_Lifetime'], categories=[1, 2], ordered=True)
df['Smoking_Status'] = df['Smoking_Status'].cat.rename_categories([
                                                                  'Yes', 'No'])

df['BMI_Category_Alt'] = pd.Categorical(
	df['BMI_Category_Alt'], categories=[1, 2, 3, 4], ordered=True)
df['BMI_Category_Alt'] = df['BMI_Category_Alt'].cat.rename_categories(
	['Underweight', 'Normal Weight', 'Overweight', 'Obese'])

df['Exercise_Past_30_Days'] = pd.Categorical(
	df['Exercise_Past_30_Days'], categories=[1, 2], ordered=True)
df['Exercise_Past_30_Days'] = df['Exercise_Past_30_Days'].cat.rename_categories([
                                                                                'Yes', 'No'])

df['Coronary_Heart_Disease'] = pd.Categorical(
	df['Coronary_Heart_Disease'], categories=[1, 2], ordered=True)
df['Coronary_Heart_Disease'] = df['Coronary_Heart_Disease'].cat.rename_categories([
                                                                                  'Yes', 'No'])

df['General_Health_Status'] = pd.Categorical(
	df['General_Health_Status'], categories=[1, 2, 3, 4, 5], ordered=True)
df['General_Health_Status'] = df['General_Health_Status'].cat.rename_categories(
	['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])


# BMI_Value as BMI_Category / 100
if 'BMI_Category' in df.columns:
	df['BMI_Value'] = df['BMI_Category'] / 100

# Initialize Age_Group_5yr as categorical with labels (like R code)
age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
              "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
df['Age_Group_5yr'] = pd.Categorical(
	df['Age_Group_5yr'], categories=range(1, 14), ordered=True)
df['Age'] = df['Age_Group_5yr'].cat.rename_categories(age_labels)


# Strict data cleaning: Remove records with missing or invalid Income_Categories
df = df[df['Income_Categories'].notnull()]
df = df[df['Income_Categories'].astype(
	str).str.strip().str.lower().isin(['', 'na', 'n/a']) == False]

df = df[df['Income_Level'].notnull()]
df = df[df['Income_Level'].astype(str).str.strip(
).str.lower().isin(['', 'na', 'n/a']) == False]

# Map INCOME3 (Income level) to descriptive labels as per codebook


def map_income_level(val):
	try:
		v = float(val)
	except:
		return "Not asked or Missing"
	if v == 1:
		return "Less than $10,000"
	elif v == 2:
		return "Less than $15,000 ($10,000 to < $15,000)"
	elif v == 3:
		return "Less than $20,000 ($15,000 to < $20,000)"
	elif v == 4:
		return "Less than $25,000 ($20,000 to < $25,000)"
	elif v == 5:
		return "Less than $35,000 ($25,000 to < $35,000)"
	elif v == 6:
		return "Less than $50,000 ($35,000 to < $50,000)"
	elif v == 7:
		return "Less than $75,000 ($50,000 to < $75,000)"
	elif v == 8:
		return "Less than $100,000 ($75,000 to < $100,000)"
	elif v == 9:
		return "Less than $150,000 ($100,000 to < $150,000)"
	elif v == 10:
		return "Less than $200,000 ($150,000 to < $200,000)"
	elif v == 11:
		return "$200,000 or more"
	elif v == 77:
		return "Don't know/Not sure"
	elif v == 99:
		return "Refused"
	else:
		return "Not asked or Missing"


if 'Income_Level' in df.columns:
	df['Income_Level'] = df['Income_Level'].apply(map_income_level)

# Initialize Income_Categories as categorical with detailed codebook mapping


def map_income_category(val):
	try:
		v = float(val)
	except:
		return "Don't know/Not sure/Missing"
	if v in [1, 1.0, 2, 2.0]:
		return "Less than $15,000"
	elif v in [3, 3.0, 4, 4.0]:
		return "$15,000 to < $25,000"
	elif v in [5, 5.0]:
		return "$25,000 to < $35,000"
	elif v in [6, 6.0]:
		return "$35,000 to < $50,000"
	elif v in [7, 7.0, 8, 8.0]:
		return "$50,000 to < $100,000"
	elif v in [9, 9.0, 10, 10.0]:
		return "$100,000 to < $200,000"
	elif v in [11, 11.0]:
		return "$200,000 or more"
	elif v in [77, 77.0, 99, 99.0]:
		return "Don't know/Not sure/Missing"
	else:
		return "Don't know/Not sure/Missing"


df['Income_Categories'] = df['Income_Categories'].apply(map_income_category)

# Print first five records where Income_Categories has a value
print(df.head())

# Print number of records after cleaning
print(f"Number of records after cleaning Household_Income_Category: {len(df)}")


# Median imputation for BMI_Value if missing percentage is between 5% and 20%
if 'BMI_Value' in df.columns:
	missing_bmi = df['BMI_Value'].isnull().sum()
	missing_pct_bmi = df['BMI_Value'].isnull().mean() * 100
	if 5 <= missing_pct_bmi <= 20:
		bmi_median = df['BMI_Value'].median(skipna=True)
		df['BMI_Value'] = df['BMI_Value'].fillna(bmi_median)
		print(
			f"Imputed {missing_bmi} missing values in BMI_Value with median ({bmi_median:.2f})")


# Impute missing BMI_Category_Alt based on BMI_Value (after variable creation)
if 'BMI_Category_Alt' in df.columns and 'BMI_Value' in df.columns:
	missing_idx = df['BMI_Category_Alt'].isnull() & df['BMI_Value'].notnull()
	if missing_idx.sum() > 0:
		bmi_vals = df.loc[missing_idx, 'BMI_Value']
		bmi_cat = pd.cut(
			bmi_vals,
			bins=[-np.inf, 18.5, 24.9, 29.9, np.inf],
			labels=["Underweight", "Normal Weight", "Overweight", "Obese"],
			right=True
		)
		df.loc[missing_idx, 'BMI_Category_Alt'] = bmi_cat
		print(
			f"Imputed {missing_idx.sum()} missing BMI_Category_Alt values based on BMI_Value")


# Remove rows where General_Health_Status is null or empty (after all variable creation and imputation)
if 'General_Health_Status' in df.columns:
	df = df[df['General_Health_Status'].notnull()]
	df = df[df['General_Health_Status'].astype(str).str.strip() != '']


# Remove rows with impossible BMI values (<10 or >70) after all variable creation and imputation
if 'BMI_Value' in df.columns:
	rows_before = len(df)
	bmi_outliers = (df['BMI_Value'] < 10) | (df['BMI_Value'] > 70)
	n_outliers = bmi_outliers.sum()
	if n_outliers > 0:
		df = df[~bmi_outliers]
		print(f"Removed {n_outliers} rows with impossible BMI values (<10 or >70)")

# Remove rows where Age is above 69 years
if 'Age' in df.columns:
	rows_before = len(df)
	valid_age_groups = ["18-24", "25-29", "30-34", "35-39",
                     "40-44", "45-49", "50-54", "55-59", "60-64", "65-69"]
	df = df[df['Age'].isin(valid_age_groups)]
	print(f"Removed {rows_before - len(df)} rows with Age above 69 years")

# Remove rows where Employment_Status is not available
if 'Employment_Status' in df.columns:
	rows_before = len(df)
	valid_emp_status = ['Employed', 'Self-employed',
                     'Out of work (>1yr)', 'Out of work (<1yr)', 'Homemaker', 'Student', 'Retired', 'Unable to work']
	df = df[df['Employment_Status'].isin(valid_emp_status)]
	print(
		f"Removed {rows_before - len(df)} rows with unavailable Employment_Status")

# Remove rows where Education_Level is not available
if 'Education_Level' in df.columns:
	rows_before = len(df)
	valid_edu_levels = ['Never attended', 'Grades 1-8', 'Grades 9-11',
                     'Grade 12/GED', 'Some college', 'College graduate']
	df = df[df['Education_Level'].isin(valid_edu_levels)]
	print(
		f"Removed {rows_before - len(df)} rows with unavailable Education_Level")

if 'Own_or_Rent_Home' in df.columns:
	rows_before = len(df)
	valid_home_status = ['Own', 'Rent', 'Other']
	df = df[df['Own_or_Rent_Home'].isin(valid_home_status)]
	print(
		f"Removed {rows_before - len(df)} rows with unavailable Own_or_Rent_Home")

if 'Primary_Insurance' in df.columns:
	rows_before = len(df)
	df = df[df['Primary_Insurance'].notnull()]
	valid_insurance = [
		'Employer/Union Plan', 'Private Plan', 'Medicare', 'Medigap', 'Medicaid', 'CHIP',
		'Military/VA', 'Indian Health Service', 'State Plan', 'Other Government'
	]
	df = df[df['Primary_Insurance'].isin(valid_insurance)]
	print(
		f"Removed {rows_before - len(df)} rows with unavailable Primary_Insurance")

# if 'Personal_Doctor' in df.columns:
# 	rows_before = len(df)
# 	valid_personal_doc = ['Yes, only one', 'More than one', 'None']
# 	df = df[df['Personal_Doctor'].isin(valid_personal_doc)]
# 	print(f"Removed {rows_before - len(df)} rows with unavailable Personal_Doctor")

print(f"Number of records after final cleaning: {len(df)}")

selected_features = [
	'Education_Level',
	'Employment_Status',
	'Own_or_Rent_Home',
	'Primary_Insurance',
	'Sex',
	'Age',
	'BMI_Value',
	'Income_Categories',
	'Income_Level',
	'BMI_Category_Alt',
	'Diabetes_Status',
	'Smoking_Status',
	'Exercise_Past_30_Days',
	'Coronary_Heart_Disease',

	'Depression',
	'COPD',
	'Heart_Attack_History',
	'CVD_Stroke_History',
	'Asthma_Ever',
	'Marital_Status',
	'Personal_Doctor',
	'Could_Not_Afford_Doctor',
	'Last_Checkup',
	'Life_Satisfaction',
	'Food_Insecurity'
]
features = [col for col in selected_features if col in df.columns]

# Impute missing values only for selected features
for col in features:
	if df[col].isnull().any():
		if df[col].dtype.kind in 'biufc':
			imp = SimpleImputer(strategy='median')
		else:
			imp = SimpleImputer(strategy='most_frequent')
		df[col] = imp.fit_transform(df[[col]]).ravel()


# Print missing value percentage for each variable
print('\nMissing value percentage for each variable:')
missing_percent = df.isnull().mean() * 100
print(missing_percent.sort_values(ascending=False))
