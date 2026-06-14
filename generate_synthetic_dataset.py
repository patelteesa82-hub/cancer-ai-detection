#!/usr/bin/env python3
"""
ONCONEXIS Synthetic Biomedical Dataset Generator
=================================================

Generates a synthetic biomedical dataset with 50,000 fictional patients
for AI prototype development and educational research purposes.

IMPORTANT DISCLAIMER:
- All records are completely synthetic and fictional
- No real patient data is included
- Not intended for clinical use
- For AI/ML prototype development and educational research only

Author: AI Research Team
Date: 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_dataset(n_patients=50000):
    """
    Generate synthetic biomedical dataset with medical plausibility.
    
    Parameters:
    -----------
    n_patients : int
        Number of synthetic patients to generate (default: 50,000)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing synthetic patient records
    """
    
    print("=" * 80)
    print("ONCONEXIS SYNTHETIC BIOMEDICAL DATASET GENERATOR")
    print("=" * 80)
    print(f"\nGenerating {n_patients:,} synthetic patient records...")
    print("Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("\nDISCLAIMER: This is synthetic data for research/educational purposes only")
    print("-" * 80)
    
    data = {}
    
    # 1. Basic Demographics
    print("\n[1/11] Generating demographic data...")
    data['Patient_ID'] = [f'SYN{i:08d}' for i in range(1, n_patients + 1)]
    data['Age'] = np.random.randint(18, 91, n_patients)
    data['Sex'] = np.random.choice(['Male', 'Female'], n_patients, p=[0.48, 0.52])
    
    # 2. Lifestyle Factors with Age Correlation
    print("[2/11] Generating lifestyle factors...")
    # Smoking status - increase with age (realistic pattern)
    smoking_probs = np.where(
        data['Age'] < 30, [0.60, 0.25, 0.15],
        np.where(
            data['Age'] < 60, [0.30, 0.50, 0.20],
            [0.20, 0.60, 0.20]
        )
    )
    data['Smoking_Status'] = np.array([
        np.random.choice(['Never', 'Former', 'Current'], p=probs)
        for probs in smoking_probs
    ])
    
    # Alcohol consumption
    data['Alcohol_Consumption'] = np.random.choice(
        ['None', 'Moderate', 'Heavy'], 
        n_patients,
        p=[0.40, 0.45, 0.15]
    )
    
    # 3. Family History and Previous Cancer
    print("[3/11] Generating family history and cancer history...")
    # Family history increases slightly with age
    family_history_prob = 0.15 + (data['Age'] - 18) * 0.003
    data['Family_History_Cancer'] = np.random.binomial(
        1, 
        np.clip(family_history_prob, 0, 0.35),
        n_patients
    ).astype(str)
    data['Family_History_Cancer'] = data['Family_History_Cancer'].map({'1': 'Yes', '0': 'No'})
    
    # Previous cancer history (increases with age)
    prev_cancer_prob = (data['Age'] - 18) * 0.003
    data['Previous_Cancer_History'] = np.random.binomial(
        1,
        np.clip(prev_cancer_prob, 0, 0.20),
        n_patients
    ).astype(str)
    data['Previous_Cancer_History'] = data['Previous_Cancer_History'].map({'1': 'Yes', '0': 'No'})
    
    # 4. Symptoms (multiple can be present)
    print("[4/11] Generating symptom data...")
    symptoms = [
        'Persistent_Cough', 'Weight_Loss', 'Fatigue', 
        'Blood_In_Stool', 'Blood_In_Urine', 'Difficulty_Swallowing',
        'Chronic_Pain', 'Breast_Lump', 'Abnormal_Bleeding', 'Persistent_Fever'
    ]
    
    # Base symptom probability increases with age and risk factors
    for symptom in symptoms:
        base_prob = 0.08 + (data['Age'] - 18) * 0.001
        data[symptom] = np.random.binomial(1, np.clip(base_prob, 0.05, 0.25), n_patients)
    
    # 5. Biomarkers - CEA and CYFRA21-1 (elevated with symptoms)
    print("[5/11] Generating biomarker data...")
    symptom_count = data[symptoms].sum(axis=1)
    
    # CEA (normal: 0-5 ng/mL, elevated: >5)
    cea_base = np.random.lognormal(mean=0.5, sigma=0.8, size=n_patients)
    cea_elevation = symptom_count * 1.5  # Elevated with more symptoms
    data['CEA'] = cea_base + cea_elevation
    data['CEA'] = np.clip(data['CEA'], 0.1, 100)  # Realistic range
    
    # CYFRA21-1 (tumor marker, normal: <3.3 ng/mL)
    # Elevated in lung cancer and other cancers
    cyfra_base = np.random.lognormal(mean=-0.5, sigma=0.9, size=n_patients)
    cyfra_elevation = (data['Smoking_Status'] == 'Current') * 2.0
    cyfra_elevation += symptom_count * 0.8
    data['CYFRA21_1'] = cyfra_base + cyfra_elevation
    data['CYFRA21_1'] = np.clip(data['CYFRA21_1'], 0.1, 50)
    
    # 6. Genetic/Molecular Markers
    print("[6/11] Generating genetic markers...")
    # EGFR Status (higher in smokers, adenocarcinoma)
    egfr_prob = 0.15 + (data['Smoking_Status'] == 'Former') * 0.05 + (data['Smoking_Status'] == 'Current') * 0.10
    data['EGFR_Status'] = np.random.binomial(
        1,
        np.clip(egfr_prob, 0.05, 0.35),
        n_patients
    ).astype(str)
    data['EGFR_Status'] = data['EGFR_Status'].map({'1': 'Positive', '0': 'Negative'})
    
    # KRAS Status (associated with colorectal and lung cancers)
    kras_prob = 0.20 + (data['CEA'] > 5) * 0.15 + (data['Age'] > 60) * 0.05
    data['KRAS_Status'] = np.random.binomial(
        1,
        np.clip(kras_prob, 0.05, 0.40),
        n_patients
    ).astype(str)
    data['KRAS_Status'] = data['KRAS_Status'].map({'1': 'Positive', '0': 'Negative'})
    
    # APC Status (colorectal cancer marker)
    apc_prob = 0.15 + (data['Family_History_Cancer'] == 'Yes') * 0.15 + (data['CEA'] > 5) * 0.10
    data['APC_Status'] = np.random.binomial(
        1,
        np.clip(apc_prob, 0.05, 0.35),
        n_patients
    ).astype(str)
    data['APC_Status'] = data['APC_Status'].map({'1': 'Positive', '0': 'Negative'})
    
    # ctDNA Status (circulating tumor DNA, advanced disease marker)
    ctdna_prob = 0.10 + (data['CEA'] > 10) * 0.20 + (data['Smoking_Status'] == 'Current') * 0.08
    data['ctDNA_Status'] = np.random.binomial(
        1,
        np.clip(ctdna_prob, 0.02, 0.40),
        n_patients
    ).astype(str)
    data['ctDNA_Status'] = data['ctDNA_Status'].map({'1': 'Positive', '0': 'Negative'})
    
    # 7. Calculate Cancer Risk Score (0-100)
    print("[7/11] Calculating cancer risk scores...")
    risk_score = np.zeros(n_patients)
    
    # Age factor
    risk_score += (data['Age'] - 18) * 0.3
    
    # Smoking factor
    risk_score += (data['Smoking_Status'] == 'Former') * 8
    risk_score += (data['Smoking_Status'] == 'Current') * 15
    
    # Alcohol factor
    risk_score += (data['Alcohol_Consumption'] == 'Moderate') * 3
    risk_score += (data['Alcohol_Consumption'] == 'Heavy') * 8
    
    # Family history
    risk_score += (data['Family_History_Cancer'] == 'Yes') * 12
    
    # Previous cancer
    risk_score += (data['Previous_Cancer_History'] == 'Yes') * 20
    
    # Symptom burden
    risk_score += symptom_count * 3
    
    # Biomarker elevation
    risk_score += (data['CEA'] > 5) * 10
    risk_score += (data['CYFRA21_1'] > 3.3) * 10
    
    # Genetic markers
    risk_score += (data['EGFR_Status'] == 'Positive') * 8
    risk_score += (data['KRAS_Status'] == 'Positive') * 10
    risk_score += (data['APC_Status'] == 'Positive') * 8
    risk_score += (data['ctDNA_Status'] == 'Positive') * 15
    
    # Normalize to 0-100 scale
    data['Cancer_Risk_Score'] = np.clip(risk_score, 0, 100)
    
    # 8. Determine Risk Category
    print("[8/11] Classifying risk categories...")
    data['Risk_Category'] = pd.cut(
        data['Cancer_Risk_Score'],
        bins=[0, 30, 60, 100],
        labels=['Low', 'Moderate', 'High'],
        include_lowest=True
    ).astype(str)
    
    # 9. Calculate Organ-Specific Cancer Probabilities
    print("[9/11] Calculating organ-specific cancer probabilities...")
    
    # Lung Cancer Probability
    lung_prob = (
        (data['Smoking_Status'] != 'Never') * 0.25 +
        (data['CYFRA21_1'] > 3.3) * 0.20 +
        (data['CEA'] > 5) * 0.10 +
        (data['EGFR_Status'] == 'Positive') * 0.15 +
        (data['Persistent_Cough'] == 1) * 0.15 +
        (data['Age'] > 60) * 0.08 +
        (data['ctDNA_Status'] == 'Positive') * 0.12
    )
    data['Lung_Cancer_Probability'] = np.clip(lung_prob * 100, 0, 95)
    
    # Colorectal Cancer Probability
    crc_prob = (
        (data['CEA'] > 5) * 0.25 +
        (data['KRAS_Status'] == 'Positive') * 0.20 +
        (data['APC_Status'] == 'Positive') * 0.25 +
        (data['Family_History_Cancer'] == 'Yes') * 0.15 +
        (data['Blood_In_Stool'] == 1) * 0.30 +
        (data['Age'] > 50) * 0.08 +
        (data['ctDNA_Status'] == 'Positive') * 0.12
    )
    data['Colorectal_Cancer_Probability'] = np.clip(crc_prob * 100, 0, 95)
    
    # Breast Cancer Probability (higher in females)
    breast_prob = (
        (data['Sex'] == 'Female') * 0.15 +
        (data['Breast_Lump'] == 1) * 0.50 +
        (data['Family_History_Cancer'] == 'Yes') * 0.20 +
        (data['Abnormal_Bleeding'] == 1) * 0.15 +
        (data['Age'] > 40) * 0.10 +
        (data['Alcohol_Consumption'] == 'Heavy') * 0.08 +
        (data['ctDNA_Status'] == 'Positive') * 0.10
    )
    data['Breast_Cancer_Probability'] = np.clip(breast_prob * 100, 0, 90)
    
    # Pancreatic Cancer Probability
    pancreatic_prob = (
        (data['Smoking_Status'] != 'Never') * 0.15 +
        (data['CEA'] > 10) * 0.20 +
        (data['Chronic_Pain'] == 1) * 0.15 +
        (data['Weight_Loss'] == 1) * 0.20 +
        (data['Difficulty_Swallowing'] == 1) * 0.10 +
        (data['Age'] > 60) * 0.12 +
        (data['Family_History_Cancer'] == 'Yes') * 0.15 +
        (data['ctDNA_Status'] == 'Positive') * 0.15
    )
    data['Pancreatic_Cancer_Probability'] = np.clip(pancreatic_prob * 100, 0, 85)
    
    # Liver Cancer Probability
    liver_prob = (
        (data['Alcohol_Consumption'] == 'Heavy') * 0.25 +
        (data['CEA'] > 5) * 0.15 +
        (data['Chronic_Pain'] == 1) * 0.12 +
        (data['Age'] > 60) * 0.10 +
        (data['ctDNA_Status'] == 'Positive') * 0.15
    )
    data['Liver_Cancer_Probability'] = np.clip(liver_prob * 100, 0, 80)
    
    # Prostate Cancer Probability (males only)
    prostate_prob = np.zeros(n_patients)
    male_mask = data['Sex'] == 'Male'
    prostate_prob[male_mask] = (
        0.15 +  # Base age/sex risk
        (data['Age'][male_mask] > 50) * 0.20 +
        (data['CEA'][male_mask] > 5) * 0.10 +
        (data['Family_History_Cancer'][male_mask] == 'Yes') * 0.20 +
        (data['ctDNA_Status'][male_mask] == 'Positive') * 0.12
    )
    data['Prostate_Cancer_Probability'] = np.clip(prostate_prob * 100, 0, 85)
    
    # 10. Create DataFrame
    print("[10/11] Creating DataFrame...")
    df = pd.DataFrame(data)
    
    # 11. Round numerical columns for realism
    print("[11/11] Finalizing dataset...")
    df['CEA'] = df['CEA'].round(2)
    df['CYFRA21_1'] = df['CYFRA21_1'].round(2)
    df['Cancer_Risk_Score'] = df['Cancer_Risk_Score'].round(1)
    
    for col in ['Lung_Cancer_Probability', 'Colorectal_Cancer_Probability', 
                'Breast_Cancer_Probability', 'Pancreatic_Cancer_Probability',
                'Liver_Cancer_Probability', 'Prostate_Cancer_Probability']:
        df[col] = df[col].round(2)
    
    return df

def calculate_correlation_summary(df):
    """
    Calculate and return correlation summary for key risk factors.
    """
    summary = {
        'Total Records': len(df),
        'Sex Distribution': df['Sex'].value_counts().to_dict(),
        'Age Statistics': {
            'Mean': f"{df['Age'].mean():.1f}",
            'Median': f"{df['Age'].median():.1f}",
            'Min': int(df['Age'].min()),
            'Max': int(df['Age'].max())
        },
        'Smoking Distribution': df['Smoking_Status'].value_counts().to_dict(),
        'Alcohol Distribution': df['Alcohol_Consumption'].value_counts().to_dict(),
        'Family History Cancer': df['Family_History_Cancer'].value_counts().to_dict(),
        'Previous Cancer History': df['Previous_Cancer_History'].value_counts().to_dict(),
        'Risk Category Distribution': df['Risk_Category'].value_counts().to_dict(),
        'Risk Score Statistics': {
            'Mean': f"{df['Cancer_Risk_Score'].mean():.1f}",
            'Median': f"{df['Cancer_Risk_Score'].median():.1f}",
            'Min': f"{df['Cancer_Risk_Score'].min():.1f}",
            'Max': f"{df['Cancer_Risk_Score'].max():.1f}"
        },
        'Biomarker Statistics': {
            'CEA': {
                'Mean': f"{df['CEA'].mean():.2f}",
                'Median': f"{df['CEA'].median():.2f}",
                'Elevated (>5)': int((df['CEA'] > 5).sum())
            },
            'CYFRA21_1': {
                'Mean': f"{df['CYFRA21_1'].mean():.2f}",
                'Median': f"{df['CYFRA21_1'].median():.2f}",
                'Elevated (>3.3)': int((df['CYFRA21_1'] > 3.3).sum())
            }
        },
        'Genetic Marker Positive Rates': {
            'EGFR': f"{(df['EGFR_Status'] == 'Positive').sum() / len(df) * 100:.1f}%",
            'KRAS': f"{(df['KRAS_Status'] == 'Positive').sum() / len(df) * 100:.1f}%",
            'APC': f"{(df['APC_Status'] == 'Positive').sum() / len(df) * 100:.1f}%",
            'ctDNA': f"{(df['ctDNA_Status'] == 'Positive').sum() / len(df) * 100:.1f}%"
        },
        'Average Cancer Probabilities': {
            'Lung': f"{df['Lung_Cancer_Probability'].mean():.2f}%",
            'Colorectal': f"{df['Colorectal_Cancer_Probability'].mean():.2f}%",
            'Breast': f"{df['Breast_Cancer_Probability'].mean():.2f}%",
            'Pancreatic': f"{df['Pancreatic_Cancer_Probability'].mean():.2f}%",
            'Liver': f"{df['Liver_Cancer_Probability'].mean():.2f}%",
            'Prostate': f"{df['Prostate_Cancer_Probability'].mean():.2f}%"
        }
    }
    return summary

if __name__ == "__main__":
    # Generate synthetic dataset
    df = generate_synthetic_dataset(n_patients=50000)
    
    # Generate correlation summary
    summary = calculate_correlation_summary(df)
    
    # Save CSV file
    csv_filename = 'ONCONEXIS_Synthetic_Dataset_50k.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n✓ Dataset saved: {csv_filename}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  File size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display summary statistics
    print("\n" + "=" * 80)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 80)
    
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    print(f"  {subkey}:")
                    for k, v in subvalue.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {subkey}: {subvalue}")
        else:
            print(f"\n{key}: {value}")
    
    print("\n" + "=" * 80)
    print("Dataset generation complete!")
    print("=" * 80)
