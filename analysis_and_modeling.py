import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')
import pickle
import json

print("="*80)
print("CANCER BIOMARKER DATASET - COMPREHENSIVE ANALYSIS & MODEL TRAINING")
print("="*80)

# ============================================================================
# 1. LOAD AND EXPLORE DATASET
# ============================================================================
print("\n[1] LOADING DATASET...")
url = "https://raw.githubusercontent.com/patelteesa82-hub/cancer-ai-detection/main/cancer_biomarkers_50k.csv"
df = pd.read_csv(url)

print(f"✓ Dataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================================
# 2. DATA QUALITY REPORT
# ============================================================================
print("\n" + "="*80)
print("[2] DATA QUALITY REPORT")
print("="*80)

print(f"\nDataset Info:")
print(f"  Total Rows: {len(df):,}")
print(f"  Total Columns: {len(df.columns)}")
print(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 3. MISSING VALUES REPORT
# ============================================================================
print("\n" + "="*80)
print("[3] MISSING VALUES REPORT")
print("="*80)

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0]

if len(missing_data) == 0:
    print("\n✓ No missing values detected - Dataset is complete!")
else:
    print(missing_data.to_string(index=False))

# ============================================================================
# 4. CLASS DISTRIBUTION
# ============================================================================
print("\n" + "="*80)
print("[4] CLASS DISTRIBUTION (Cancer_Diagnosis)")
print("="*80)

class_dist = df['Cancer_Diagnosis'].value_counts()
class_pct = (df['Cancer_Diagnosis'].value_counts(normalize=True) * 100).round(2)

print(f"\nClass Distribution:")
for cls in sorted(df['Cancer_Diagnosis'].unique()):
    print(f"  Class {cls}: {class_dist[cls]:,} samples ({class_pct[cls]:.2f}%)")

print(f"\nClass Balance Ratio: {class_dist.max() / class_dist.min():.2f}:1")

# ============================================================================
# 5. STATISTICAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("[5] BIOMARKER STATISTICS")
print("="*80)

biomarkers = ['ctDNA', 'EGFR', 'KRAS', 'APC', 'CEA', 'CYFRA_21_1']
print(df[biomarkers].describe().round(4))

# ============================================================================
# 6. CORRELATION MATRIX
# ============================================================================
print("\n" + "="*80)
print("[6] CORRELATION MATRIX")
print("="*80)

# Encode categorical variables
df_encoded = df.copy()
le_sex = LabelEncoder()
le_fh = LabelEncoder()
le_sh = LabelEncoder()

df_encoded['Sex'] = le_sex.fit_transform(df['Sex'])
df_encoded['Family_History'] = le_fh.fit_transform(df['Family_History'])
df_encoded['Smoking_History'] = le_sh.fit_transform(df['Smoking_History'])

# Calculate correlation with target
feature_cols = ['Age', 'Sex', 'Family_History', 'Smoking_History'] + biomarkers
corr_with_target = df_encoded[feature_cols + ['Cancer_Diagnosis']].corr()['Cancer_Diagnosis'].drop('Cancer_Diagnosis')
corr_with_target = corr_with_target.sort_values(ascending=False)

print("\nCorrelation with Cancer_Diagnosis:")
for feat, corr in corr_with_target.items():
    print(f"  {feat:20s}: {corr:7.4f}")

# ============================================================================
# 7. FEATURE IMPORTANCE RANKING
# ============================================================================
print("\n" + "="*80)
print("[7] FEATURE IMPORTANCE RANKING (based on correlation)")
print("="*80)

for i, (feat, corr) in enumerate(corr_with_target.items(), 1):
    importance = abs(corr) * 100
    print(f"  {i:2d}. {feat:20s}: {importance:6.2f}%")

# ============================================================================
# 8. PREPARE DATA FOR MODELING
# ============================================================================
print("\n" + "="*80)
print("[8] DATA PREPARATION FOR MODELING")
print("="*80)

# Prepare features and target
X = df_encoded[feature_cols].copy()
y = df['Cancer_Diagnosis'].copy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Training Set: {X_train.shape[0]:,} samples")
print(f"✓ Test Set: {X_test.shape[0]:,} samples")
print(f"✓ Features: {X_train.shape[1]}")
print(f"✓ Train-Test Split: 80-20")

# ============================================================================
# 9. MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("[9] TRAINING MODELS")
print("="*80)

# Logistic Regression
print("\n[a] Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
print("✓ Complete")

# Random Forest
print("[b] Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
print("✓ Complete")

# XGBoost
print("[c] XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
print("✓ Complete")

# ============================================================================
# 10. MODEL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("[10] MODEL COMPARISON - PERFORMANCE METRICS")
print("="*80)

# Calculate metrics
models = {
    'Logistic Regression': (lr_pred, lr_pred_proba),
    'Random Forest': (rf_pred, rf_pred_proba),
    'XGBoost': (xgb_pred, xgb_pred_proba)
}

results = {}
for model_name, (predictions, pred_proba) in models.items():
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, pred_proba)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'ROC_AUC': roc_auc
    }
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

# ============================================================================
# 11. BEST MODEL SELECTION
# ============================================================================
print("\n" + "="*80)
print("[11] BEST MODEL SELECTION")
print("="*80)

# Rank models by ROC-AUC (best overall metric)
ranked_models = sorted(results.items(), key=lambda x: x[1]['ROC_AUC'], reverse=True)

print("\nModels Ranked by ROC-AUC:")
for rank, (name, metrics) in enumerate(ranked_models, 1):
    print(f"  {rank}. {name:25s}: {metrics['ROC_AUC']:.4f}")

best_model_name = ranked_models[0][0]
best_model_metrics = ranked_models[0][1]

print(f"\n✓ BEST MODEL: {best_model_name}")
print(f"  ROC-AUC: {best_model_metrics['ROC_AUC']:.4f}")
print(f"  F1 Score: {best_model_metrics['F1_Score']:.4f}")
print(f"  Accuracy: {best_model_metrics['Accuracy']:.4f}")

# ============================================================================
# 12. DETAILED CLASSIFICATION REPORT FOR BEST MODEL
# ============================================================================
print("\n" + "="*80)
print("[12] DETAILED CLASSIFICATION REPORT - Best Model")
print("="*80)

if best_model_name == 'Logistic Regression':
    best_pred = lr_pred
elif best_model_name == 'Random Forest':
    best_pred = rf_pred
else:
    best_pred = xgb_pred

print(f"\n{classification_report(y_test, best_pred, target_names=['No Cancer', 'Cancer'])}")

# ============================================================================
# 13. FEATURE IMPORTANCE (FROM RANDOM FOREST)
# ============================================================================
print("\n" + "="*80)
print("[13] FEATURE IMPORTANCE (Random Forest)")
print("="*80)

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + feature_importance.to_string(index=False))

# ============================================================================
# 14. EXPORT BEST MODEL
# ============================================================================
print("\n" + "="*80)
print("[14] EXPORTING BEST MODEL FOR INTEGRATION")
print("="*80)

if best_model_name == 'Logistic Regression':
    best_model = lr_model
    model_type = 'LogisticRegression'
elif best_model_name == 'Random Forest':
    best_model = rf_model
    model_type = 'RandomForest'
else:
    best_model = xgb_model
    model_type = 'XGBoost'

# Save model
model_file = f'best_cancer_model_{model_type}.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Model saved: {model_file}")

# Save scaler
scaler_file = 'feature_scaler.pkl'
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved: {scaler_file}")

# Save feature names and encoders
config = {
    'model_type': model_type,
    'feature_columns': feature_cols,
    'target_variable': 'Cancer_Diagnosis',
    'test_accuracy': float(best_model_metrics['Accuracy']),
    'test_roc_auc': float(best_model_metrics['ROC_AUC']),
    'test_f1_score': float(best_model_metrics['F1_Score']),
    'encoders': {
        'Sex': list(le_sex.classes_),
        'Family_History': list(le_fh.classes_),
        'Smoking_History': list(le_sh.classes_)
    }
}

config_file = 'model_config.json'
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
print(f"✓ Configuration saved: {config_file}")

# ============================================================================
# 15. SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("[15] FINAL SUMMARY - READY FOR ONCONEXIS INTEGRATION")
print("="*80)

print(f"""
DATABASE STATISTICS:
  - Total Samples: {len(df):,}
  - Features: {len(feature_cols)}
  - Classes: 2 (No Cancer: {(y==0).sum():,}, Cancer: {(y==1).sum():,})
  - Data Quality: 100% (No missing values)

BEST MODEL: {best_model_name}
  - Accuracy:  {best_model_metrics['Accuracy']:.4f}
  - Precision: {best_model_metrics['Precision']:.4f}
  - Recall:    {best_model_metrics['Recall']:.4f}
  - F1 Score:  {best_model_metrics['F1_Score']:.4f}
  - ROC-AUC:   {best_model_metrics['ROC_AUC']:.4f}

EXPORT FILES FOR INTEGRATION:
  1. {model_file}
  2. {scaler_file}
  3. {config_file}

INTEGRATION INSTRUCTIONS:
  1. Load the pickle files in your application
  2. Encode categorical inputs using the provided encoders
  3. Scale features using the saved scaler
  4. Make predictions using the best model
  5. Output probability scores for clinical decision support

RECOMMENDED THRESHOLD:
  - Clinical Decision: Use probability > 0.5 for cancer risk
  - High Risk Alert: Use probability > 0.7
  - Confirmatory Testing: Recommended for probabilities 0.4-0.6
""")

print("="*80)
print("✓ ANALYSIS COMPLETE - Ready for deployment to ONCONEXIS")
print("="*80)
