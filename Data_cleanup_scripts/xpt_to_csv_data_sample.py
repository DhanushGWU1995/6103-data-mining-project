import pandas as pd
import os
import sys

def check_available_variables(xpt_path, target_vars):
    """
    Quick check to see which variables are available in the XPT file without loading all data
    
    Args:
        xpt_path (str): Path to the input XPT file
        target_vars (list): List of variables to check for
    """
    try:
        # Read just the first few rows to get column names
        df_sample = pd.read_sas(xpt_path, format='xport', chunksize=1)
        chunk = next(df_sample)
        available_cols = list(chunk.columns)
        
        available_vars = [var for var in target_vars if var in available_cols]
        missing_vars = [var for var in target_vars if var not in available_cols]
        
        print(f"Quick check - Variables found: {len(available_vars)} out of {len(target_vars)}")
        if available_vars:
            print("Available BRFSS variables:")
            for var in available_vars[:10]:  # Show first 10
                print(f"  ✓ {var}")
            if len(available_vars) > 10:
                print(f"  ... and {len(available_vars) - 10} more")
        
        return available_vars, missing_vars
    except Exception as e:
        print(f"Error checking variables: {e}")
        return [], target_vars

def xpt_to_csv(xpt_path, csv_path, extract_specific_vars=True):
    """
    Convert XPT (SAS transport) file to CSV with error handling
    
    Args:
        xpt_path (str): Path to the input XPT file
        csv_path (str): Path to the output CSV file
        extract_specific_vars (bool): If True, extract only BRFSS 2015 variables from the mapping table
    """
    # Define the BRFSS 2024 variables - Updated based on codebook analysis
    brfss_variables = [
        # 'SEQNO',         # Patient ID (sequence number)
        # '_AGEG5YR',      # Age (age groups in 5-year intervals)
        # 'DIABETE4',      # Diabetes (diabetes status) - UPDATED FROM DIABETE3 TO DIABETE4
        # 'FLUSHOT7',      # Flu vaccination
        # 'SMOKE100',      # Smoked at least 100 cigarettes in lifetime
        # 'ALCDAY4',       # Alcohol consumption days (2024 version)  
        # 'DRNK3GE5',      # Binge drinking frequency
        # 'MAXDRNKS',      # Most drinks on single occasion
        # '_RFSMOK3',      # Smoking (_RFSMOK3: 1=current smoker, 0=former/never)
        # 'SMOKDAY2',      # Smoking (base variable for _RFSMOK3)
        # '_BMI5',         # Obesity/BMI (BMI categories)
        # '_BMI5CAT',      # BMI categories (alternative)
        # '_RFBMI5',       # BMI risk factor
        # '_TOTINDA',      # Exercise (1=meets aerobic guidelines, 0=no)
        # 'EXERANY2',      # Exercise in Past 30 Days (main physical activity variable in 2024)
        # 'DIFFWALK',      # Difficulty Walking or Climbing Stairs
        # 'CVDSTRK3',      # Previous Heart Problems (CVD/stroke)
        # 'CVDINFR4',      # Previous Heart Problems (heart attack)
        # 'CVDCRHD4',      # Previous Heart Problems (coronary/angina) - UPDATED FROM CHCSCNCR TO CVDCRHD4
        # 'GENHLTH',       # General Health Status - ADDED KEY HEALTH VARIABLE
        # 'MENTHLTH',      # Stress Level (poor mental health days)
        # 'POORHLTH',      # Poor physical health days
        # 'WEIGHT2',       # Weight - ADDED PHYSICAL MEASUREMENT
        # 'HEIGHT3',       # Height - ADDED PHYSICAL MEASUREMENT  
        # '_STATE',        # Country/State (state code)
        # '_PSU',          # Region-Specific: primary sampling unit
        # '_RFHLTH',       # Region-Specific: health prevalence by region
        # '_INCOMG1',      # Region-Specific: income level
        # 'INCOME3',       # Income (household income categories) - UPDATED FROM INCOME2 TO INCOME3
        # '_SEX',          # Sex (redundant with SEXVAR, but included for completeness)

        # 'SEQNO', # Patient ID (sequence number)
        '_AGEG5YR',      # Age Group (5-year intervals)
        'DIABETE4',      # Diabetes Status (updated from DIABETE3 to DIABETE4)
        'FLUSHOT7',      # Flu Vaccination
        'SMOKE100',      # Smoked 100 Cigarettes in Lifetime
        'ALCDAY4',      # Alcohol Days Per Month
        'DRNK3GE5',      # Binge Drinking Episodes
        'MAXDRNKS',      # Max Drinks Single Occasion
        '_RFSMOK3',      # Smoking Status
        'SMOKDAY2',      # Smoking Frequency
        '_BMI5',         # BMI Category
        '_BMI5CAT',     # BMI Category (Alternative)
        '_RFBMI5',      # BMI Risk Factor
        '_TOTINDA',     # Exercise Guidelines Met
        'EXERANY2',     # Exercise in Past 30 Days
        'DIFFWALK',    # Difficulty Walking Stairs
        'CVDSTRK3',    # CVD Stroke History
        'CVDINFR4',    # Heart Attack History
        'CVDCRHD4',    # Coronary Heart Disease (Updated from CHCSCNCR to CVDCRHD4)
        'GENHLTH',     # General Health Status (Added new variable)
        'MENTHLTH',    # Mental Health Days
        'POORHLTH',    # Poor Physical Health Days
        'WEIGHT2',     # Weight (Pounds) - Added new variable
        'HEIGHT3',     # Height (Feet/Inches) - Added new variable
        '_STATE',      # State Code
        '_PSU',        # Primary Sampling Unit
        '_RFHLTH',     # Regional Health Prevalence
        '_INCOMG1',    # Income Level
        'INCOME3',     # Income Categories - Added new variable
        '_SEX',       # Sex
        # --- Added below: Only those not already present ---
        # Demographics & Socioeconomic Status
        'EDUCA',     # Education_Level
        'RENTHOM1',  # Own_or_Rent_Home
        'EMPLOY1',   # Employment_Status
        'MARITAL',   # Marital_Status
        'VETERAN3',  # Veteran_Status
        'CHILDREN',  # Num_Children
        # '_IMPRACE',  # Imputed_Race
        # '_HISPANC',  # Hispanic_Origin
        # '_RACE',     # Race
        # Health Access & Utilization
        
        'PERSDOC3',  # Personal_Doctor
        'MEDCOST1',  # Could_Not_Afford_Doctor
        'CHECKUP1',  # Last_Checkup
        # Health Behaviors
        # 'USENOW3',  # Smokeless_Tobacco_Use
        # 'ECIGNOW3',  # E_Cig_Use
        # 'AVEDRNK4',  # Avg_Drinks_Per_Day
        # 'SSBSUGR2',  # Sugary_Soda_Freq
        # 'SSBFRUT3',  # Sugary_FruitDrink_Freq
        # Chronic Conditions & Health Status
        'PHYSHLTH',  # Physical_Health_NotGood_Days
        'ASTHMA3',   # Asthma_Ever
        'ASTHNOW',   # Asthma_Current
        # 'CHCSCNC1',  # Skin_Cancer
        # 'CHCOCNC1',  # Other_Cancer
        'CHCCOPD3',  # COPD
        'ADDEPEV3',  # Depression
        # 'CHCKDNY2',  # Kidney_Disease
        # 'HAVARTH4',  # Arthritis
        # Disability & Functional Status
        # 'DEAF',  # Deaf_or_Hearing_Difficulty
        # 'BLIND',  # Blind_or_Visual_Difficulty
        # 'DECIDE',  # Difficulty_Concentrating
        # 'DIFFDRES',  # Difficulty_Dressing_Bathing
        # 'DIFFALON',  # Difficulty_Doing_Errands_Alone
        # Social Determinants & Adversity
        'LSATISFY',  # Life_Satisfaction
        'EMTSUPRT',  # Emotional_Support
        # 'SDLONELY',  # Loneliness
        # 'SDHEMPLY',  # Lost_Employment
        # 'FOODSTMP',  # Received_Food_Stamps
        'SDHFOOD1',  # Food_Insecurity
        'SDHBILLS',  # Unable_to_Pay_Bills
        # 'SDHUTILS',  # Utility_Shutoff_Threat
        # 'SDHTRNSP',  # Transportation_Insecurity
        # 'ACEDEPRS',  # ACE_Depressed_Household
        # 'ACEDRINK',  # ACE_Alcoholic_Household
        # 'ACEDRUGS',  # ACE_Drug_Abuse_Household
        # 'ACEPRISN',  # ACE_Prison_Household
        # 'ACEDIVRC',  # ACE_Parents_Divorced
        # 'ACEPUNCH',  # ACE_Parental_Violence
        # 'ACEHURT1',  # ACE_Parental_Physical_Abuse
        # 'ACESWEAR',  # ACE_Parental_Emotional_Abuse
        # 'ACETOUCH',  # ACE_Sexual_Abuse
        # 'ACETTHEM',  # ACE_Sexual_Abuse_Them
        # 'ACEHVSEX',  # ACE_Forced_Sex
        # 'ACEADSAF',  # ACE_Adult_Made_Safe
        # 'ACEADNED',  # ACE_Adult_Met_Needs
        # Other
        'PREGNANT',  # Pregnancy_Status
        # 'RMVTETH4',  # Teeth_Removed
        'LASTDEN4',  # Last_Dental_Visit

        'PRIMINS2',  # Primary_Insurance
    ]
    try:
        # Check if input file exists
        if not os.path.exists(xpt_path):
            raise FileNotFoundError(f"Input file '{xpt_path}' not found")
        
        # Check file extension
        if not xpt_path.lower().endswith('.xpt'):
            print(f"Warning: '{xpt_path}' does not have .xpt extension")
        
        # Read the .xpt (SAS transport) file
        # Try different approaches for reading XPT files
        df = None
        
        # Method 1: Try with format='xport'
        try:
            df = pd.read_sas(xpt_path, format='xport')
            print(f"Successfully read {xpt_path} using format='xport'")
        except Exception as e1:
            print(f"Failed to read with format='xport': {e1}")
            
            # Method 2: Try without specifying format (let pandas auto-detect)
            try:
                df = pd.read_sas(xpt_path)
                print(f"Successfully read {xpt_path} using auto-detection")
            except Exception as e2:
                print(f"Failed to read with auto-detection: {e2}")
                
                # Method 3: Try reading as binary and using xport library if available
                try:
                    import xport
                    with open(xpt_path, 'rb') as f:
                        library = xport.load(f)
                        # Get the first dataset (assuming single dataset)
                        dataset_name = list(library.keys())[0]
                        df = library[dataset_name]
                        df = pd.DataFrame(df)
                    print(f"Successfully read {xpt_path} using xport library")
                except ImportError:
                    print("xport library not available. Install with: pip install xport")
                    raise Exception("Unable to read XPT file with any method")
                except Exception as e3:
                    print(f"Failed to read with xport library: {e3}")
                    raise Exception("Unable to read XPT file with any method")
        
        if df is None:
            raise Exception("Failed to read the XPT file")
        
        # Display basic info about the dataset
        print(f"Dataset shape: {df.shape}")
        print(f"Total columns available: {len(df.columns)}")
        
        # Filter to extract only the specific BRFSS variables if requested
        if extract_specific_vars:
            # Find which variables from our list are actually present in the dataset
            available_vars = [var for var in brfss_variables if var in df.columns]
            missing_vars = [var for var in brfss_variables if var not in df.columns]
            
            print(f"\nBRFSS 2024 Variables found in dataset ({len(available_vars)} out of {len(brfss_variables)}):")
            for var in available_vars:
                print(f"  ✓ {var}")
            
            if missing_vars:
                print(f"\nBRFSS 2024 Variables NOT found in dataset ({len(missing_vars)}):")
                for var in missing_vars:
                    print(f"  ✗ {var}")
            
            # Filter the dataframe to include only the available BRFSS variables
            if available_vars:
                df_filtered = df[available_vars].copy()
                print(f"\nFiltered dataset shape: {df_filtered.shape}")
                
                # Create readable column name mapping - Updated for 2024 BRFSS variables
                column_mapping = {
                    # 'SEQNO': 'Patient_ID',
                    '_AGEG5YR': 'Age_Group_5yr',
                    'DIABETE4': 'Diabetes_Status',  # Updated from DIABETE3 to DIABETE4
                    'FLUSHOT7': 'Flu_Vaccination',
                    'SMOKE100': 'Smoked_100_Cigarettes_Lifetime',
                    'ALCDAY4': 'Alcohol_Days_Per_Month',
                    'DRNK3GE5': 'Binge_Drinking_Episodes',
                    'MAXDRNKS': 'Max_Drinks_Single_Occasion',
                    '_RFSMOK3': 'Smoking_Status',
                    'SMOKDAY2': 'Smoking_Frequency',
                    '_BMI5': 'BMI_Category',
                    '_BMI5CAT': 'BMI_Category_Alt',
                    '_RFBMI5': 'BMI_Risk_Factor',
                    '_TOTINDA': 'Exercise_Guidelines_Met',
                    'EXERANY2': 'Exercise_Past_30_Days',
                    'DIFFWALK': 'Difficulty_Walking_Stairs',
                    'CVDSTRK3': 'CVD_Stroke_History',
                    'CVDINFR4': 'Heart_Attack_History',
                    'CVDCRHD4': 'Coronary_Heart_Disease',  # Updated from CHCSCNCR to CVDCRHD4
                    'GENHLTH': 'General_Health_Status',  # Added new variable
                    'MENTHLTH': 'Mental_Health_Days',
                    'POORHLTH': 'Poor_Physical_Health_Days',
                    'WEIGHT2': 'Weight_Pounds',  # Added new variable
                    'HEIGHT3': 'Height_Feet_Inches',  # Added new variable
                    '_STATE': 'State_Code',
                    '_PSU': 'Primary_Sampling_Unit',
                    '_RFHLTH': 'Regional_Health_Prevalence',
                    '_INCOMG1': 'Income_Level',
                    'INCOME3': 'Income_Categories',    # Added new variable
                    '_SEX': 'Sex',
                    # --- Added below: Only those not already present ---
                    # Demographics & Socioeconomic Status
                    'EDUCA': 'Education_Level',
                    'RENTHOM1': 'Own_or_Rent_Home',
                    'EMPLOY1': 'Employment_Status',
                    'MARITAL': 'Marital_Status',
                    'VETERAN3': 'Veteran_Status',
                    'CHILDREN': 'Num_Children',
                    # '_IMPRACE': 'Imputed_Race',
                    # '_HISPANC': 'Hispanic_Origin',
                    # '_RACE': 'Race',
                    # Health Access & Utilization
                    
                    'PERSDOC3': 'Personal_Doctor',
                    'MEDCOST1': 'Could_Not_Afford_Doctor',
                    'CHECKUP1': 'Last_Checkup',
                    # Health Behaviors
                    # 'USENOW3': 'Smokeless_Tobacco_Use',
                    # 'ECIGNOW3': 'E_Cig_Use',
                    # 'AVEDRNK4': 'Avg_Drinks_Per_Day',
                    # 'SSBSUGR2': 'Sugary_Soda_Freq',
                    # 'SSBFRUT3': 'Sugary_FruitDrink_Freq',
                    # Chronic Conditions & Health Status
                    'PHYSHLTH': 'Physical_Health_NotGood_Days',
                    'ASTHMA3': 'Asthma_Ever',
                    'ASTHNOW': 'Asthma_Current',
                    # 'CHCSCNC1': 'Skin_Cancer',
                    # 'CHCOCNC1': 'Other_Cancer',
                    'CHCCOPD3': 'COPD',
                    'ADDEPEV3': 'Depression',
                    # 'CHCKDNY2': 'Kidney_Disease',
                    # 'HAVARTH4': 'Arthritis',
                    # Disability & Functional Status
                    # 'DEAF': 'Deaf_or_Hearing_Difficulty',
                    # 'BLIND': 'Blind_or_Visual_Difficulty',
                    # 'DECIDE': 'Difficulty_Concentrating',
                    # 'DIFFDRES': 'Difficulty_Dressing_Bathing',
                    # 'DIFFALON': 'Difficulty_Doing_Errands_Alone',
                    # Social Determinants & Adversity
                    'LSATISFY': 'Life_Satisfaction',
                    'EMTSUPRT': 'Emotional_Support',
                    # 'SDLONELY': 'Loneliness',
                    # 'SDHEMPLY': 'Lost_Employment',
                    # 'FOODSTMP': 'Received_Food_Stamps',
                    'SDHFOOD1': 'Food_Insecurity',
                    'SDHBILLS': 'Unable_to_Pay_Bills',
                    # 'SDHUTILS': 'Utility_Shutoff_Threat',
                    # 'SDHTRNSP': 'Transportation_Insecurity',
                    # 'ACEDEPRS': 'ACE_Depressed_Household',
                    # 'ACEDRINK': 'ACE_Alcoholic_Household',
                    # 'ACEDRUGS': 'ACE_Drug_Abuse_Household',
                    # 'ACEPRISN': 'ACE_Prison_Household',
                    # 'ACEDIVRC': 'ACE_Parents_Divorced',
                    # 'ACEPUNCH': 'ACE_Parental_Violence',
                    # 'ACEHURT1': 'ACE_Parental_Physical_Abuse',
                    # 'ACESWEAR': 'ACE_Parental_Emotional_Abuse',
                    # 'ACETOUCH': 'ACE_Sexual_Abuse',
                    # 'ACETTHEM': 'ACE_Sexual_Abuse_Them',
                    # 'ACEHVSEX': 'ACE_Forced_Sex',
                    # 'ACEADSAF': 'ACE_Adult_Made_Safe',
                    # 'ACEADNED': 'ACE_Adult_Met_Needs',
                    # Other
                    'PREGNANT': 'Pregnancy_Status',
                    # 'RMVTETH4': 'Teeth_Removed',
                    'LASTDEN4': 'Last_Dental_Visit',
                    'PRIMINS2': 'Primary_Insurance',
                }
                
                # Rename columns to readable names
                df_filtered = df_filtered.rename(columns=column_mapping)
                print(f"Renamed columns to readable format")
                print(f"New column names: {list(df_filtered.columns)}")
                df = df_filtered
            else:
                print("\nWarning: No BRFSS 2024 variables found in the dataset!")
                print("Available columns in dataset:")
                for i, col in enumerate(df.columns):
                    print(f"  {i+1:3d}. {col}")
        else:
            print(f"All columns: {list(df.columns)}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Write to CSV
        df.to_csv(csv_path, index=False)
        print(f"Successfully converted '{xpt_path}' to '{csv_path}'")
        
        return df
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error converting XPT to CSV: {e}")
        return None

# Example usage with error handling:
if __name__ == "__main__":
    # Allow command line arguments for input and output files
    # Usage: python script.py [input.xpt] [output.csv]
    if len(sys.argv) >= 2:
        xpt_file = sys.argv[1]
        csv_file = sys.argv[2] if len(sys.argv) >= 3 else xpt_file.replace('.xpt', '_extracted.csv').replace('.XPT', '_extracted.csv')
    else:
        # Default files - you can change these as needed
        xpt_file = 'LLCP2024.XPT '  # Note: file has trailing space
        csv_file = 'BRFSS_2024_Readable_Columns.csv'

    print(f"Input file: '{xpt_file}'")
    print(f"Output file: '{csv_file}'")
    
    if os.path.exists(xpt_file):
        # Extract only BRFSS 2024 variables by default
        result = xpt_to_csv(xpt_file, csv_file, extract_specific_vars=True)
        if result is not None:
            print(f"\nConversion successful! Check {csv_file}")
            print(f"Extracted BRFSS 2024 variables from the mapping table.")
        else:
            print("Conversion failed!")
    else:
        print(f"XPT file '{xpt_file}' not found in current directory.")
        print("Available files in current directory:")
        files = [f for f in os.listdir('.') if f.endswith(('.xpt', '.sas7bdat', '.csv'))]
        if files:
            for file in files:
                print(f"  - {file}")
        else:
            print("  No SAS/XPT/CSV files found")