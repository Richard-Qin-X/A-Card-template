import os
import pandas as pd

COLUMN_NAMES = [
    'status_checking', 'duration_months', 'credit_history', 'purpose',
    'credit_amount', 'savings_account', 'employment_since', 'installment_rate',
    'personal_status_sex', 'other_debtors', 'residence_since', 'property',
    'age', 'other_installments', 'housing', 'existing_credits',
    'job', 'num_dependents', 'telephone', 'foreign_worker', 'target'
]

GERMAN_CREDIT_MAPPING = {
    'status_checking': {
        'A11': '< 0 DM', 
        'A12': '0 <= ... < 200 DM', 
        'A13': '>= 200 DM / salary assignments for at least 1 year', 
        'A14': 'no checking account'
    },
    'credit_history': {
        'A30': 'no credits taken/ all credits paid back duly',
        'A31': 'all credits at this bank paid back duly',
        'A32': 'existing credits paid back duly till now',
        'A33': 'delay in paying off in the past',
        'A34': 'critical account/ other credits existing (not at this bank)'
    },
    'purpose': {
        'A40': 'car (new)', 'A41': 'car (used)', 'A42': 'furniture/equipment',
        'A43': 'radio/television', 'A44': 'domestic appliances', 'A45': 'repairs',
        'A46': 'education', 'A47': '(vacation - does not exist?)', 'A48': 'retraining',
        'A49': 'business', 'A410': 'others'
    },
    'savings_account': {
        'A61': '< 100 DM', 'A62': '100 <= ... < 500 DM',
        'A63': '500 <= ... < 1000 DM', 'A64': '>= 1000 DM',
        'A65': 'unknown/ no savings account'
    },
    'employment_since': {
        'A71': 'unemployed', 'A72': '< 1 year',
        'A73': '1 <= ... < 4 years', 'A74': '4 <= ... < 7 years',
        'A75': '>= 7 years'
    },
    'personal_status_sex': {
        'A91': 'male: divorced/separated',
        'A92': 'female: divorced/separated/married',
        'A93': 'male: single',
        'A94': 'male: married/widowed',
        'A95': 'female: single'
    },
    'other_debtors': {
        'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'
    },
    'property': {
        'A121': 'real estate',
        'A122': 'building society savings agreement/ life insurance',
        'A123': 'car or other',
        'A124': 'unknown / no property'
    },
    'other_installments': {
        'A141': 'bank', 'A142': 'stores', 'A143': 'none'
    },
    'housing': {
        'A151': 'rent', 'A152': 'own', 'A153': 'for free'
    },
    'job': {
        'A171': 'unemployed/ unskilled - non-resident',
        'A172': 'unskilled - resident',
        'A173': 'skilled employee / official',
        'A174': 'management/ self-employed/ highly qualified employee/ officer'
    },
    'telephone': {
        'A191': 'none', 'A192': 'yes, registered under the customers name'
    },
    'foreign_worker': {
        'A201': 'yes', 'A202': 'no'
    }
}

def  load_and_clean_data(file_path):
    df = pd.read_csv(file_path, sep=' ', header=None, names=COLUMN_NAMES)

    df['target'] = df['target'].map({1: 0, 2: 1})

    for col, mapping in GERMAN_CREDIT_MAPPING.items():
        if col in df.columns:
            unmapped_vals = set(df[col].unique()) - set(mapping.keys())
            if unmapped_vals:
                print(f"Warning: Column {col} has unmapped values: {unmapped_vals}")
            
            df[col] = df[col].map(mapping)

    categorical_cols = list(GERMAN_CREDIT_MAPPING.keys())
    df[categorical_cols] = df[categorical_cols].astype('category')

    return df

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)).replace('src', '')
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    FILE_PATH = os.path.join(DATA_DIR, 'german.data')

    df_clean = load_and_clean_data(FILE_PATH)
    print(df_clean.head())
    print(df_clean['target'].value_counts(normalize=True))
    df_clean.to_parquet(PROJECT_ROOT + '/data/german_clean.parquet', index=False)