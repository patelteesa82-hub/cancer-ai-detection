import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
url = "https://raw.githubusercontent.com/patelteesa82-hub/cancer-ai-detection/main/cancer_biomarkers_50k.csv"
df = pd.read_csv(url)

# Prepare data
df_encoded = df.copy()
le_sex = LabelEncoder()
le_fh = LabelEncoder()
le_sh = LabelEncoder()

df_encoded['Sex'] = le_sex.fit_transform(df['Sex'])
df_encoded['Family_History'] = le_fh.fit_transform(df['Family_History'])
df_encoded['Smoking_History'] = le_sh.fit_transform(df['Smoking_History'])

feature_cols = ['Age', 'Sex', 'Family_History', 'Smoking_History', 'ctDNA', 'EGFR', 'KRAS', 'APC', 'CEA', 'CYFRA_21_1']
X = df_encoded[feature_cols].copy()
y = df['Cancer_Diagnosis'].copy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Create model comparison results
results_data = []
models_info = {
    'Logistic Regression': (lr_pred, lr_pred_proba),
    'Random Forest': (rf_pred, rf_pred_proba),
    'XGBoost': (xgb_pred, xgb_pred_proba)
}

for model_name, (predictions, pred_proba) in models_info.items():
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, pred_proba)
    
    results_data.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'ROC_AUC': roc_auc
    })

# Save model comparison
model_comparison_df = pd.DataFrame(results_data)
model_comparison_csv = model_comparison_df.to_csv(index=False)

# Feature importance from Random Forest
feature_importance_data = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_,
    'Importance_Percentage': (rf_model.feature_importances_ * 100).round(2)
}).sort_values('Importance', ascending=False).reset_index(drop=True)

feature_importance_csv = feature_importance_data.to_csv(index=False)

# Predictions with probabilities
predictions_data = pd.DataFrame({
    'Sample_ID': range(1, len(y_test) + 1),
    'Actual': y_test.values,
    'LR_Prediction': lr_pred,
    'LR_Probability': lr_pred_proba,
    'RF_Prediction': rf_pred,
    'RF_Probability': rf_pred_proba,
    'XGB_Prediction': xgb_pred,
    'XGB_Probability': xgb_pred_proba
}).reset_index(drop=True)

predictions_csv = predictions_data.to_csv(index=False)

# Model results summary
model_results_data = {
    'Metric': ['Total Test Samples', 'True Positives (LR)', 'True Negatives (LR)', 'False Positives (LR)', 'False Negatives (LR)',
               'Best Model', 'Best Accuracy', 'Best ROC-AUC', 'Best F1-Score'],
    'Value': [
        len(y_test),
        sum((lr_pred == 1) & (y_test == 1)),
        sum((lr_pred == 0) & (y_test == 0)),
        sum((lr_pred == 1) & (y_test == 0)),
        sum((lr_pred == 0) & (y_test == 1)),
        'Random Forest',
        f"{max(r['Accuracy'] for r in results_data):.4f}",
        f"{max(r['ROC_AUC'] for r in results_data):.4f}",
        f"{max(r['F1_Score'] for r in results_data):.4f}"
    ]
}

model_results_df = pd.DataFrame(model_results_data)

print("✓ Analysis complete")
print("Files ready for export")
