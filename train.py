"""
train.py

End-to-end training script for the Telco Customer Churn predictor.
Run with: python train.py

Trains three classifiers (Logistic Regression, Random Forest, XGBoost),
selects the best by ROC-AUC, saves model artefacts, and generates all
EDA and SHAP visualisations to plots/.
"""

import warnings
warnings.filterwarnings('ignore')

import joblib
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = Path('models')
PLOTS_DIR = Path('plots')
RANDOM_STATE = 42
TEST_SIZE = 0.2

DATA_URL = (
    'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/'
    'master/data/Telco-Customer-Churn.csv'
)

MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Data loading and cleaning
# ---------------------------------------------------------------------------
print('Loading dataset…')
df = pd.read_csv(DATA_URL)
print(f'  Loaded {len(df):,} rows, {df.shape[1]} columns')

# TotalCharges arrives as object due to spaces; coerce and drop 11 bad rows
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
print(f'  After dropping NaN TotalCharges: {len(df):,} rows')

df['Churn'] = (df['Churn'] == 'Yes').astype(int)
df.drop(columns=['customerID'], inplace=True)

churn_rate = df['Churn'].mean() * 100
print(f'  Churn rate: {churn_rate:.1f}%')

# ---------------------------------------------------------------------------
# 2. Exploratory analysis — 4 charts saved to plots/
# ---------------------------------------------------------------------------
print('\nGenerating EDA plots…')

# 01 — Churn distribution pie chart
fig, ax = plt.subplots(figsize=(6, 6))
counts = df['Churn'].value_counts()
labels = [f'Retained\n{counts[0]:,}', f'Churned\n{counts[1]:,}']
colors = ['#2E75B6', '#C0392B']
ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
       startangle=90, textprops={'fontsize': 12})
ax.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
fig.savefig(PLOTS_DIR / '01_churn_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# 02 — Churn rate by contract type
fig, ax = plt.subplots(figsize=(8, 5))
contract_churn = df.groupby('Contract')['Churn'].mean() * 100
contract_churn.sort_values(ascending=False).plot(
    kind='bar', ax=ax, color='#2E75B6', edgecolor='white'
)
ax.set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Contract Type', fontsize=11)
ax.set_ylabel('Churn Rate (%)', fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.1f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='bottom', fontsize=10)
fig.savefig(PLOTS_DIR / '02_churn_by_contract.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# 03 — Tenure histogram coloured by churn
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df[df['Churn'] == 0]['tenure'], bins=30, alpha=0.6,
        color='#2E75B6', label='Retained')
ax.hist(df[df['Churn'] == 1]['tenure'], bins=30, alpha=0.6,
        color='#C0392B', label='Churned')
ax.set_title('Tenure Distribution by Churn Status', fontsize=14, fontweight='bold')
ax.set_xlabel('Tenure (months)', fontsize=11)
ax.set_ylabel('Customer count', fontsize=11)
ax.legend(fontsize=11)
fig.savefig(PLOTS_DIR / '03_churn_by_tenure.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# 04 — Monthly charges KDE by churn status
fig, ax = plt.subplots(figsize=(9, 5))
df[df['Churn'] == 0]['MonthlyCharges'].plot.kde(
    ax=ax, color='#2E75B6', label='Retained', linewidth=2
)
df[df['Churn'] == 1]['MonthlyCharges'].plot.kde(
    ax=ax, color='#C0392B', label='Churned', linewidth=2
)
ax.set_title('Monthly Charges Distribution by Churn Status', fontsize=14, fontweight='bold')
ax.set_xlabel('Monthly Charges ($)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.legend(fontsize=11)
fig.savefig(PLOTS_DIR / '04_monthly_charges_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print('  Saved plots 01–04')

# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
print('\nPreprocessing…')

# Binary Yes/No columns → 1/0
binary_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
]
for col in binary_cols:
    if col == 'gender':
        df[col] = (df[col] == 'Male').astype(int)
    else:
        df[col] = (df[col] == 'Yes').astype(int)

# One-hot encode nominal categoricals (drop_first avoids multicollinearity)
df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', 'InternetService'],
                    drop_first=True)

# Separate target
y = df.pop('Churn')
X = df.copy()

# Scale numeric columns in-place
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

feature_names = list(X.columns)
print(f'  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features')

# Stratified split preserves churn ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f'  Train: {len(X_train):,}  Test: {len(X_test):,}')

# ---------------------------------------------------------------------------
# 4. Model training
# ---------------------------------------------------------------------------
print('\nTraining models…')

models = {
    'Logistic Regression': LogisticRegression(
        random_state=RANDOM_STATE, max_iter=1000
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss',
        verbosity=0
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'model': model,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
    }
    print(f'  {name}: ROC-AUC={results[name]["ROC-AUC"]:.4f}  F1={results[name]["F1"]:.4f}')

# Comparison table
metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
comparison = pd.DataFrame(
    {name: {m: results[name][m] for m in metric_cols} for name in results}
).T
print('\nModel comparison:')
print(comparison.to_string(float_format='{:.4f}'.format))

# ---------------------------------------------------------------------------
# 5. Select best model by ROC-AUC
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda n: results[n]['ROC-AUC'])
best_model = results[best_name]['model']
print(f'\nBest model: {best_name} (ROC-AUC={results[best_name]["ROC-AUC"]:.4f})')

# ---------------------------------------------------------------------------
# 6. Save model artefacts
# ---------------------------------------------------------------------------
joblib.dump(best_model, MODEL_DIR / 'best_model.pkl')
joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
joblib.dump(feature_names, MODEL_DIR / 'feature_names.pkl')
(MODEL_DIR / 'best_model_name.txt').write_text(best_name)
print(f'  Saved to {MODEL_DIR}/')

# ---------------------------------------------------------------------------
# 7. MLOps versioning — timestamped copy + model registry
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
versioned_path = MODEL_DIR / f'best_model_{timestamp}.pkl'
joblib.dump(best_model, versioned_path)

registry_path = MODEL_DIR / 'model_registry.csv'
registry_row = {
    'timestamp': timestamp,
    'model_name': best_name,
    'roc_auc': round(results[best_name]['ROC-AUC'], 4),
    'f1': round(results[best_name]['F1'], 4),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'model_file': str(versioned_path),
}

if registry_path.exists():
    registry = pd.read_csv(registry_path)
else:
    registry = pd.DataFrame(columns=registry_row.keys())

registry = pd.concat([registry, pd.DataFrame([registry_row])], ignore_index=True)
registry.to_csv(registry_path, index=False)

print(f'\nMLOps registry entry:')
print(pd.DataFrame([registry_row]).to_string(index=False))
print(f'Versioned model: {versioned_path}')

# ---------------------------------------------------------------------------
# 8. SHAP analysis on best model
# ---------------------------------------------------------------------------
print('\nRunning SHAP analysis…')

X_test_arr = X_test.values
X_train_arr = X_train.values

if best_name == 'Logistic Regression':
    explainer = shap.LinearExplainer(best_model, X_train_arr, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_test_arr)
else:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_arr)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

# Force float64 ndarray — LinearExplainer can return Python floats in some SHAP versions
shap_values = np.array(shap_values, dtype=np.float64)

# 05 — SHAP beeswarm summary plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.title('SHAP Feature Impact — All Test Customers', fontsize=13, fontweight='bold')
plt.savefig(PLOTS_DIR / '05_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 05_shap_summary.png')

# Identify highest and lowest churn probability customers in test set
y_proba_test = best_model.predict_proba(X_test_arr)[:, 1]
high_risk_idx = int(np.argmax(y_proba_test))
low_risk_idx = int(np.argmin(y_proba_test))

# Normalise expected_value to a plain Python float regardless of explainer type
base_val = float(
    explainer.expected_value[0]
    if isinstance(explainer.expected_value, (list, np.ndarray))
    else explainer.expected_value
)

# 06 — Waterfall for highest-probability churn customer
shap_exp_high = shap.Explanation(
    values=shap_values[high_risk_idx],
    base_values=base_val,
    data=X_test_arr[high_risk_idx],
    feature_names=feature_names,
)
fig, ax = plt.subplots(figsize=(10, 7))
shap.waterfall_plot(shap_exp_high, max_display=12, show=False)
plt.title(f'SHAP Waterfall — Highest Churn Risk Customer (p={y_proba_test[high_risk_idx]:.2f})',
          fontsize=12, fontweight='bold')
plt.savefig(PLOTS_DIR / '06_shap_waterfall_churn.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 06_shap_waterfall_churn.png')

# 07 — Waterfall for lowest-probability churn customer
shap_exp_low = shap.Explanation(
    values=shap_values[low_risk_idx],
    base_values=base_val,
    data=X_test_arr[low_risk_idx],
    feature_names=feature_names,
)
fig, ax = plt.subplots(figsize=(10, 7))
shap.waterfall_plot(shap_exp_low, max_display=12, show=False)
plt.title(f'SHAP Waterfall — Lowest Churn Risk Customer (p={y_proba_test[low_risk_idx]:.2f})',
          fontsize=12, fontweight='bold')
plt.savefig(PLOTS_DIR / '07_shap_waterfall_retained.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 07_shap_waterfall_retained.png')

# 08 — Model comparison grouped bar chart
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(metric_cols))
width = 0.25
colors = ['#2E75B6', '#27AE60', '#E67E22']
for i, (name, color) in enumerate(zip(results.keys(), colors)):
    vals = [results[name][m] for m in metric_cols]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(metric_cols, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Model Comparison — All Metrics', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(y=0.8, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
fig.savefig(PLOTS_DIR / '08_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved 08_model_comparison.png')

# ---------------------------------------------------------------------------
# 9. Final summary
# ---------------------------------------------------------------------------
print('\n' + '=' * 60)
print('TRAINING COMPLETE — SUMMARY')
print('=' * 60)
print('\nAll model metrics:')
print(comparison.to_string(float_format='{:.4f}'.format))

print(f'\nWinner: {best_name}')
print(f'  ROC-AUC = {results[best_name]["ROC-AUC"]:.4f} — selected because ROC-AUC')
print('  is threshold-independent and best reflects churn detection')
print('  performance across all operating points.')

# Top 5 SHAP features by mean absolute SHAP value
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:5]
print('\nTop 5 SHAP features (mean |SHAP|):')
for rank, i in enumerate(top_idx, 1):
    print(f'  {rank}. {feature_names[i]:<40} {mean_abs_shap[i]:.4f}')

print(f'\nModel artefacts: {MODEL_DIR}/')
print(f'Plots:           {PLOTS_DIR}/')
print(f'Registry:        {registry_path}')
print('=' * 60)
