import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 50,000 cancer biomarker records
n_samples = 50000

# Generate data
data = {
    'Patient_ID': [f'PAT_{i:06d}' for i in range(1, n_samples + 1)],
    
    # Biomarkers
    'ctDNA': np.random.gamma(shape=2, scale=0.5, size=n_samples),
    'EGFR': np.random.gamma(shape=2, scale=1.2, size=n_samples),
    'KRAS': np.random.gamma(shape=1.5, scale=0.8, size=n_samples),
    'APC': np.random.exponential(scale=1.5, size=n_samples),
    'CEA': np.random.gamma(shape=2, scale=1.5, size=n_samples),
    'CYFRA_21_1': np.random.gamma(shape=2, scale=0.8, size=n_samples),
    
    # Demographics
    'Age': np.random.normal(loc=62, scale=12, size=n_samples).astype(int),
    'Sex': np.random.choice(['Male', 'Female'], size=n_samples, p=[0.55, 0.45]),
    'Family_History': np.random.choice(['No', 'Yes'], size=n_samples, p=[0.70, 0.30]),
    'Smoking_History': np.random.choice(['Never', 'Former', 'Current'], 
                                        size=n_samples, 
                                        p=[0.40, 0.35, 0.25]),
}

# Create DataFrame
df = pd.DataFrame(data)
df['Age'] = df['Age'].clip(18, 95)

# Risk classification based on biomarker patterns
def classify_risk(row):
    markers = [row['ctDNA'], row['EGFR'], row['KRAS'], row['APC'], row['CEA'], row['CYFRA_21_1']]
    elevated_count = sum(1 for m in markers if m > np.median(markers))
    if elevated_count >= 4:
        return 'High Risk'
    elif elevated_count >= 2:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df['Risk_Classification'] = df.apply(classify_risk, axis=1)
df['Cancer_Diagnosis'] = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])

# Reorder columns
columns_order = [
    'Patient_ID', 'Age', 'Sex', 'Family_History', 'Smoking_History',
    'ctDNA', 'EGFR', 'KRAS', 'APC', 'CEA', 'CYFRA_21_1',
    'Risk_Classification', 'Cancer_Diagnosis'
]
df = df[columns_order]

# Save to Excel
output_file = 'cancer_biomarkers_50k.xlsx'
df.to_excel(output_file, index=False, sheet_name='Biomarkers', engine='openpyxl')

print("="*80)
print("CANCER BIOMARKER DATASET GENERATED")
print("="*80)
print(f"✓ File: {output_file}")
print(f"✓ Total Records: {len(df):,}")
print(f"✓ Columns: {len(df.columns)}")
print(f"\nColumns:")
for col in df.columns:
    print(f"  - {col}")
print(f"\nDataset Preview (first 5 rows):")
print(df.head())
print("\n" + "="*80)
