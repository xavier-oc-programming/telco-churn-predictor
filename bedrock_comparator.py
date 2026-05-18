"""
bedrock_comparator.py

Benchmarks Amazon Bedrock (Claude Haiku 4.5) against the trained XGBoost classifier
on a 20-customer sample: 10 churned, 10 retained.

Bedrock classifies churn from a plain-English customer summary — zero task-specific
training. XGBoost uses the same preprocessed features as train.py.

Run with: python bedrock_comparator.py

# Requires AWS credentials: aws configure or environment variables
# AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
# Requires Bedrock model access enabled in AWS console for:
# us.anthropic.claude-haiku-4-5-20251001-v1:0  (Claude Haiku 4.5 cross-region profile)
# Region: us-east-1
"""

import json
import warnings
warnings.filterwarnings('ignore')

import boto3
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants (must match train.py exactly)
# ---------------------------------------------------------------------------
DATA_URL = (
    'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/'
    'master/data/Telco-Customer-Churn.csv'
)
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 10  # 10 churned + 10 retained

# ---------------------------------------------------------------------------
# 1. Reproduce preprocessing from train.py using saved scaler + feature_names
# ---------------------------------------------------------------------------
print('Loading and preprocessing data…')

df = pd.read_csv(DATA_URL)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
df.drop(columns=['customerID'], inplace=True)

# Save raw feature columns before encoding (for building plain-English summaries)
raw_df = df.copy()

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

df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', 'InternetService'],
                    drop_first=True)

y = df.pop('Churn')
X = df.copy()

# Use the saved scaler so preprocessing is byte-for-byte identical to training
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X[numeric_cols] = scaler.transform(X[numeric_cols])

# Align columns: add any columns present in feature_names but missing here
for col in feature_names:
    if col not in X.columns:
        X[col] = 0
X = X[feature_names]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
# Keep raw_df aligned to the same index for building prompts
raw_test = raw_df.loc[X_test.index]

print(f'  Test set: {len(X_test):,} customers')

# ---------------------------------------------------------------------------
# 2. Sample 20 customers: 10 churned, 10 retained
# ---------------------------------------------------------------------------
churned_idx = y_test[y_test == 1].sample(SAMPLE_SIZE, random_state=RANDOM_STATE).index
retained_idx = y_test[y_test == 0].sample(SAMPLE_SIZE, random_state=RANDOM_STATE).index
sample_idx = churned_idx.tolist() + retained_idx.tolist()

X_sample = X_test.loc[sample_idx]
y_sample = y_test.loc[sample_idx]
raw_sample = raw_test.loc[sample_idx]

# ---------------------------------------------------------------------------
# 3. XGBoost predictions on the same 20 customers
# ---------------------------------------------------------------------------
best_model = joblib.load('models/best_model.pkl')
xgb_proba = best_model.predict_proba(X_sample.values)[:, 1]
xgb_preds = (xgb_proba >= 0.5).astype(int)

# ---------------------------------------------------------------------------
# 4. Build plain-English summaries for Bedrock
# ---------------------------------------------------------------------------
def build_customer_summary(row: pd.Series) -> str:
    """Convert a raw customer row into a natural-language profile string."""
    senior = 'yes' if row.get('SeniorCitizen', 0) == 1 else 'no'
    tech = row.get('TechSupport', 'No')
    security = row.get('OnlineSecurity', 'No')
    return (
        f"Customer profile: {int(row['tenure'])} months tenure, "
        f"${row['MonthlyCharges']:.2f}/month, "
        f"{row.get('Contract', 'Unknown')} contract, "
        f"{row.get('InternetService', 'Unknown')} internet, "
        f"tech support: {tech}, "
        f"online security: {security}, "
        f"payment: {row.get('PaymentMethod', 'Unknown')}, "
        f"paperless billing: {row.get('PaperlessBilling', 'No')}, "
        f"senior citizen: {senior}."
    )

# ---------------------------------------------------------------------------
# 5. Bedrock inference — one call per customer
# ---------------------------------------------------------------------------
print('\nConnecting to Amazon Bedrock…')

try:
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
except Exception as e:
    print(f'  Failed to initialise Bedrock client: {e}')
    bedrock = None

BEDROCK_PROMPT_TEMPLATE = (
    "You are a telecom churn analyst. Based on this customer profile, "
    "predict whether this customer is likely to churn (cancel their subscription). "
    'Respond with only a JSON object with these exact keys: '
    '{{"prediction": "churn" or "retain", '
    '"confidence": "high" or "medium" or "low", '
    '"reasoning": "one sentence max"}}'
    "\n\nCustomer: {summary}"
)

bedrock_results = []
success_count = 0
fail_count = 0

for i, idx in enumerate(sample_idx):
    raw_row = raw_sample.loc[idx]
    summary = build_customer_summary(raw_row)
    prompt = BEDROCK_PROMPT_TEMPLATE.format(summary=summary)

    result = {
        'index': i,
        'ground_truth': int(y_sample.loc[idx]),
        'xgb_prediction': int(xgb_preds[i]),
        'xgb_probability': round(float(xgb_proba[i]), 3),
        'bedrock_prediction': None,
        'bedrock_confidence': None,
        'bedrock_reasoning': None,
        'summary': summary,
    }

    if bedrock is None:
        fail_count += 1
        bedrock_results.append(result)
        continue

    try:
        body = json.dumps({
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 200,
            'messages': [{'role': 'user', 'content': prompt}],
        })
        response = bedrock.invoke_model(
            modelId='us.anthropic.claude-haiku-4-5-20251001-v1:0',
            body=body,
        )
        raw_response = json.loads(response['body'].read())
        text = raw_response['content'][0]['text'].strip()
        # Strip markdown code fences if model wraps JSON in ```json ... ```
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
            text = text.strip()
        parsed = json.loads(text)

        result['bedrock_prediction'] = 1 if parsed.get('prediction') == 'churn' else 0
        result['bedrock_confidence'] = parsed.get('confidence', 'unknown')
        result['bedrock_reasoning'] = parsed.get('reasoning', '')
        success_count += 1
        print(f'  [{i+1:02d}/20] Bedrock: {parsed["prediction"]} ({parsed["confidence"]})')

    except Exception as e:
        print(f'  [{i+1:02d}/20] Bedrock call failed: {e}')
        fail_count += 1

    bedrock_results.append(result)

print(f'\nBedrock calls: {success_count} succeeded, {fail_count} failed/skipped')

# ---------------------------------------------------------------------------
# 6. Print comparison table
# ---------------------------------------------------------------------------
print('\nPrediction Comparison Table')
print('-' * 95)
header = f"{'#':>3} | {'Ground Truth':>12} | {'XGBoost':>8} | {'Bedrock':>8} | {'Confidence':>10} | Reasoning"
print(header)
print('-' * 95)

for r in bedrock_results:
    gt = 'Churn' if r['ground_truth'] == 1 else 'Retain'
    xgb = 'Churn' if r['xgb_prediction'] == 1 else 'Retain'
    bed = r['bedrock_prediction']
    bed_str = ('Churn' if bed == 1 else 'Retain') if bed is not None else 'N/A'
    conf = r['bedrock_confidence'] or 'N/A'
    reason = (r['bedrock_reasoning'] or '')[:45]
    print(f"{r['index']+1:>3} | {gt:>12} | {xgb:>8} | {bed_str:>8} | {conf:>10} | {reason}")

print('-' * 95)

# ---------------------------------------------------------------------------
# 7. Confusion matrix comparison plot — 09_bedrock_comparison.png
# ---------------------------------------------------------------------------
y_true = [r['ground_truth'] for r in bedrock_results]
y_xgb = [r['xgb_prediction'] for r in bedrock_results]
y_bedrock = [r['bedrock_prediction'] for r in bedrock_results if r['bedrock_prediction'] is not None]
y_true_bedrock = [r['ground_truth'] for r in bedrock_results if r['bedrock_prediction'] is not None]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrix Comparison — 20-Customer Sample', fontsize=14, fontweight='bold')

for ax, y_pred, title, color in [
    (axes[0], y_xgb, 'XGBoost (trained)', '#2E75B6'),
    (axes[1], y_bedrock if y_bedrock else [0] * len(y_true_bedrock or y_true),
     'Amazon Bedrock Claude Haiku (zero-shot)', '#E67E22'),
]:
    y_t = y_true if title.startswith('XGBoost') else (y_true_bedrock or y_true)
    cm = confusion_matrix(y_t, y_pred if y_pred else [0] * len(y_t))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Retain', 'Churn'])
    ax.set_yticklabels(['Retain', 'Churn'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, color='white' if cm[i, j] > cm.max() / 2 else 'black')

plt.tight_layout()
plt.savefig('plots/09_bedrock_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('\nSaved plots/09_bedrock_comparison.png')

# ---------------------------------------------------------------------------
# 8. Save raw results to JSON
# ---------------------------------------------------------------------------
with open('models/bedrock_results.json', 'w') as f:
    json.dump(bedrock_results, f, indent=2)
print('Saved models/bedrock_results.json')
