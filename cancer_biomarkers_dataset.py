import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate 50,000 cancer biomarker records
n_samples = 50000

# Generate data
data = {
    'Patient_ID': [f'PAT_{i:06d}' for i in range(1, n_samples + 1)],
    
    # Biomarkers - continuous variables (realistic ranges based on clinical cutoffs)
    'ctDNA': np.random.gamma(shape=2, scale=0.5, size=n_samples),  # copies/mL
    'EGFR': np.random.gamma(shape=2, scale=1.2, size=n_samples),  # ng/mL
    'KRAS': np.random.gamma(shape=1.5, scale=0.8, size=n_samples),  # ng/mL
    'APC': np.random.exponential(scale=1.5, size=n_samples),  # ng/mL
    'CEA': np.random.gamma(shape=2, scale=1.5, size=n_samples),  # ng/mL
    'CYFRA_21_1': np.random.gamma(shape=2, scale=0.8, size=n_samples),  # ng/mL
    
    # Demographic variables
    'Age': np.random.normal(loc=62, scale=12, size=n_samples).astype(int),
    'Sex': np.random.choice(['Male', 'Female'], size=n_samples, p=[0.55, 0.45]),
    'Family_History': np.random.choice(['No', 'Yes'], size=n_samples, p=[0.70, 0.30]),
    'Smoking_History': np.random.choice(['Never', 'Former', 'Current'], 
                                        size=n_samples, 
                                        p=[0.40, 0.35, 0.25]),
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure Age is within realistic range
df['Age'] = df['Age'].clip(18, 95)

# Add Cancer Type classification based on biomarker patterns (for context)
def classify_cancer_type(row):
    """Simple classification based on biomarker elevation patterns"""
    markers = [row['ctDNA'], row['EGFR'], row['KRAS'], row['APC'], row['CEA'], row['CYFRA_21_1']]
    elevated_count = sum(1 for m in markers if m > np.median(markers))
    
    if elevated_count >= 4:
        return 'High Risk'
    elif elevated_count >= 2:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df['Risk_Classification'] = df.apply(classify_cancer_type, axis=1)

# Add diagnosis status (binary outcome for model training)
df['Cancer_Diagnosis'] = np.random.choice([0, 1], size=n_samples, 
                                         p=[0.65, 0.35])  # 35% positive cases

# Reorder columns logically
columns_order = [
    'Patient_ID',
    'Age',
    'Sex',
    'Family_History',
    'Smoking_History',
    'ctDNA',
    'EGFR',
    'KRAS',
    'APC',
    'CEA',
    'CYFRA_21_1',
    'Risk_Classification',
    'Cancer_Diagnosis'
]

df = df[columns_order]

# Display basic statistics
print("="*80)
print("CANCER BIOMARKER DATASET - 50,000 SAMPLES")
print("="*80)
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 10 rows:")
print(df.head(10))

print(f"\n\nDataset Summary Statistics:")
print(df.describe())

print(f"\n\nCategorical Variables Distribution:")
print(f"\nSex Distribution:\n{df['Sex'].value_counts()}")
print(f"\nFamily History Distribution:\n{df['Family_History'].value_counts()}")
print(f"\nSmoking History Distribution:\n{df['Smoking_History'].value_counts()}")
print(f"\nRisk Classification Distribution:\n{df['Risk_Classification'].value_counts()}")
print(f"\nCancer Diagnosis Distribution:\n{df['Cancer_Diagnosis'].value_counts()}")

print(f"\n\nBiomarker Statistics:")
for marker in ['ctDNA', 'EGFR', 'KRAS', 'APC', 'CEA', 'CYFRA_21_1']:
    print(f"\n{marker}:")
    print(f"  Mean: {df[marker].mean():.3f}")
    print(f"  Std:  {df[marker].std():.3f}")
    print(f"  Min:  {df[marker].min():.3f}")
    print(f"  Max:  {df[marker].max():.3f}")

# Save to CSV
output_file = 'cancer_biomarkers_50k.csv'
df.to_csv(output_file, index=False)
print(f"\n{'='*80}")
print(f"✓ Dataset saved to: {output_file}")
print(f"✓ Total records: {len(df):,}")
print(f"{'='*80}\n")

# Save to Excel (optional)
try:
    df.to_excel('cancer_biomarkers_50k.xlsx', index=False, sheet_name='Biomarkers')
    print("✓ Excel file also created: cancer_biomarkers_50k.xlsx\n")
except:
    print("⚠ Excel export requires openpyxl library\n")
