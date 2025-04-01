# data preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os

def clean_numerical_values(value):
    """Convert special numerical formats to regular numbers"""
    if isinstance(value, str):
        if value.startswith('>'):
            return float(value[1:]) + 1
        elif value.startswith('<'):
            return float(value[1:]) - 1
        elif '-' in value:  # Handle ranges like "50-60"
            return float(value.split('-')[0])
    return float(value)

def load_and_clean_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Define columns
    numerical_cols = ['loan_amount', 'rate_of_interest', 'Upfront_charges', 
                     'property_value', 'income', 'LTV', 'dtir1', 'Credit_Score']
    categorical_cols = ['loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 
                       'loan_purpose', 'Credit_Worthiness', 'open_credit',
                       'business_or_commercial', 'credit_type', 
                       'co-applicant_credit_type', 'submission_of_application',
                       'Region', 'Security_Type', 'construction_type',
                       'occupancy_type', 'interest_only', 'Neg_ammortization',
                       'lump_sum_payment', 'Secured_by', 'total_units']

    # Clean numerical columns
    for col in numerical_cols:
        if col in df.columns:
            # First convert special formats
            if col == 'Credit_Score':
                df[col] = df[col].apply(clean_numerical_values)
            # Then convert to numeric, coercing any remaining errors
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill remaining NA with median
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Convert categorical columns to numerical
    df['loan_limit'] = df['loan_limit'].map({'cf': 1, 'ncf': 0})
    df['approv_in_adv'] = df['approv_in_adv'].map({'pre': 1, 'nopre': 0})
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 2, 'Sex Not Available': 3, 'Joint': 4})
    df['loan_type'] = df['loan_type'].map({'type1': 1, 'type2': 2, 'type3': 3})
    df['loan_purpose'] = df['loan_purpose'].map({'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4})
    df['Credit_Worthiness'] = df['Credit_Worthiness'].map({'l1': 1, 'l2': 0})
    df['open_credit'] = df['open_credit'].map({'opc': 1, 'nopc': 0})
    df['business_or_commercial'] = df['business_or_commercial'].map({'b/c': 1, 'nob/c': 0})
    df['credit_type'] = df['credit_type'].map({'EXP': 1, 'EQUI': 2, 'CRIF': 3, 'CIB': 4})
    df['co-applicant_credit_type'] = df['co-applicant_credit_type'].map({'CIB': 1, 'EXP': 0})
    df['submission_of_application'] = df['submission_of_application'].map({'to_inst': 1, 'not_inst': 0})
    df['Region'] = df['Region'].map({'south': 1, 'North': 2, 'central': 3, 'North-East': 4})
    df['Security_Type'] = df['Security_Type'].map({'direct': 1, 'indirect': 0})

    # Label encode remaining categorical features
    label_encoder = preprocessing.LabelEncoder()
    remaining_categorical = ['construction_type', 'occupancy_type', 'interest_only',
                            'Neg_ammortization', 'lump_sum_payment', 'Secured_by', 'total_units']
    for col in remaining_categorical:
        df[col] = label_encoder.fit_transform(df[col])

    # Drop unnecessary columns
    df = df.drop(columns=["ID", "year"])
    
    # Ensure all values are numeric
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return df

def split_data(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    df = load_and_clean_data('data/raw/Loan_Default.csv')
    X_train, X_test, y_train, y_test = split_data(df, target='Status')
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)


    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("Data preprocessing complete. Processed files saved in data/processed/")