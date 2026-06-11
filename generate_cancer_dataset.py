import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 50000

# Generate dataset
data = {
    'Sample_ID': [f'CANCER_{i:06d}' for i in range(1, n_samples + 1)],
    
    # Biomarkers - ctDNA (copy number, typically 0-1000 copies/mL)
    'ctDNA': np.random.exponential(scale=50, size=n_samples),
    
    # EGFR mutation (ng/mL, typically 0-500)
    'EGFR': np.random.gamma(shape=2, scale=20, size=n_samples),
    
    # KRAS mutation (ng/mL, typically 0-300)
    'KRAS': np.random.gamma(shape=1.5, scale=15, size=n_samples),
    
    # APC mutation (ng/mL, typically 0-400)
    'APC': np.random.gamma(shape=2, scale=25, size=n_samples),
    
    # CEA (ng/mL, normal <5, elevated >10)
    'CEA': np.random.exponential(scale=3, size=n_samples),
    
    # CYFRA 21-1 (ng/mL, normal <3.3, elevated >3.3)
    'CYFRA_21_1': np.random.exponential(scale=2, size=n_samples),
    
    # Demographics
    'Age': np.random.normal(loc=62, scale=12, size=n_samples).astype(int),
    'Sex': np.random.choice(['Male', 'Female'], size=n_samples, p=[0.55, 0.45]),
    'Family_History': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.25, 0.75]),
    'Smoking_History': np.random.choice(['Never', 'Former', 'Current'], size=n_samples, p=[0.35, 0.40, 0.25]),
    
    # Additional useful columns
    'Cancer_Type': np.random.choice(['Lung', 'Colorectal', 'Breast', 'Gastric', 'Pancreatic'], 
                                     size=n_samples, p=[0.35, 0.25, 0.20, 0.12, 0.08]),
    'Stage': np.random.choice(['I', 'II', 'III', 'IV'], size=n_samples, p=[0.15, 0.25, 0.35, 0.25]),
    'Status': np.random.choice(['Healthy', 'Early_Detection', 'Advanced'], 
                                size=n_samples, p=[0.20, 0.35, 0.45]),
    'Collection_Date': [datetime(2023, 1, 1) + timedelta(days=int(x)) 
                        for x in np.random.uniform(0, 730, n_samples)],
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure Age is within realistic range (18-95)
df['Age'] = df['Age'].clip(18, 95)

# Add some realistic correlations
# Higher biomarkers correlate with advanced cancer
advanced_mask = df['Status'] == 'Advanced'
df.loc[advanced_mask, 'ctDNA'] *= np.random.uniform(2, 5, advanced_mask.sum())
df.loc[advanced_mask, 'CEA'] *= np.random.uniform(1.5, 4, advanced_mask.sum())
df.loc[advanced_mask, 'CYFRA_21_1'] *= np.random.uniform(1.5, 4, advanced_mask.sum())

# Smoking correlates with some biomarkers
smoker_mask = df['Smoking_History'] != 'Never'
df.loc[smoker_mask, 'ctDNA'] *= np.random.uniform(1.2, 1.8, smoker_mask.sum())

# Family history correlates with some biomarkers
family_mask = df['Family_History'] == 'Yes'
df.loc[family_mask, 'KRAS'] *= np.random.uniform(1.2, 1.6, family_mask.sum())

# Round biomarkers to 2 decimal places
biomarker_cols = ['ctDNA', 'EGFR', 'KRAS', 'APC', 'CEA', 'CYFRA_21_1']
df[biomarker_cols] = df[biomarker_cols].round(2)

# Save to CSV
output_file = 'cancer_biomarkers_50k.csv'
df.to_csv(output_file, index=False)

print(f"Dataset created successfully!")
print(f"Total samples: {len(df):,}")
print(f"Output file: {output_file}")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nData types:")
print(df.dtypes)
print(f"\nBasic statistics:")
print(df[biomarker_cols].describe())
print(f"\nCancer Type distribution:")
print(df['Cancer_Type'].value_counts())
print(f"\nStatus distribution:")
print(df['Status'].value_counts())
