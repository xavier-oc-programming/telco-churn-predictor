"""
shap_to_language.py

Translates raw SHAP values for a single churn prediction into plain-English
factors that a non-technical stakeholder can understand and act on.

Imported by app.py at startup. No classes — module-level dicts and one function.

Key: feature names here must exactly match the column names produced by
train.py's preprocessing. Binary Yes/No columns (TechSupport, OnlineSecurity,
etc.) are encoded directly as 0/1 integers — their column names have NO '_Yes'
suffix. Only the pd.get_dummies columns (Contract, PaymentMethod,
InternetService) carry a category suffix.
"""

# Maps every preprocessed feature name to a human-readable label.
FEATURE_LABELS = {
    # Numeric
    'tenure':          'Customer tenure',
    'MonthlyCharges':  'Monthly charges',
    'TotalCharges':    'Total charges',
    'SeniorCitizen':   'Senior citizen',

    # One-hot: Contract (baseline = Month-to-month)
    'Contract_One year': 'One-year contract',
    'Contract_Two year': 'Two-year contract',

    # One-hot: InternetService (baseline = DSL)
    'InternetService_Fiber optic': 'Fiber optic internet',
    'InternetService_No':          'No internet service',

    # One-hot: PaymentMethod (baseline = Bank transfer (automatic))
    'PaymentMethod_Credit card (automatic)': 'Credit card (automatic) payment',
    'PaymentMethod_Electronic check':        'Electronic check payment',
    'PaymentMethod_Mailed check':            'Mailed check payment',

    # Binary 0/1 columns — no '_Yes' suffix (encoded directly in train.py)
    'TechSupport':      'Tech support subscription',
    'OnlineSecurity':   'Online security subscription',
    'OnlineBackup':     'Online backup subscription',
    'DeviceProtection': 'Device protection subscription',
    'StreamingTV':      'Streaming TV subscription',
    'StreamingMovies':  'Streaming movies subscription',
    'PaperlessBilling': 'Paperless billing',
    'Partner':          'Has a partner',
    'Dependents':       'Has dependents',
    'PhoneService':     'Has phone service',
    'MultipleLines':    'Multiple phone lines',
    'gender':           'Gender',
}

# Maps each feature name to (text_when_shap_positive, text_when_shap_negative).
# shap > 0 → feature pushes toward churn (risk).
# shap < 0 → feature pushes away from churn (protective).
DIRECTION_TEMPLATES = {
    'tenure': (
        'short tenure increases churn risk',
        'long tenure reduces churn risk',
    ),
    'MonthlyCharges': (
        'high monthly charges increase churn risk',
        'low monthly charges reduce churn risk',
    ),
    'TotalCharges': (
        'low cumulative spend increases churn risk',
        'high cumulative spend reduces churn risk',
    ),
    'SeniorCitizen': (
        'senior citizen status correlates with higher churn risk',
        'non-senior status correlates with lower churn risk',
    ),

    # Contract
    'Contract_One year': (
        'absence of a one-year contract increases churn risk',
        'one-year contract reduces churn risk',
    ),
    'Contract_Two year': (
        'absence of a two-year contract increases churn risk',
        'two-year contract strongly reduces churn risk',
    ),

    # Internet service
    'InternetService_Fiber optic': (
        'fiber optic internet correlates with higher churn',
        'non-fiber internet correlates with lower churn',
    ),
    'InternetService_No': (
        'no internet service correlates with higher churn in this segment',
        'having internet service reduces churn risk',
    ),

    # Payment method
    'PaymentMethod_Credit card (automatic)': (
        'absence of automatic credit card payment increases churn risk',
        'automatic credit card payment reduces churn risk',
    ),
    'PaymentMethod_Electronic check': (
        'electronic check payment correlates with higher churn',
        'non-electronic-check payment reduces churn risk',
    ),
    'PaymentMethod_Mailed check': (
        'mailed check payment correlates with higher churn',
        'non-mailed-check payment reduces churn risk',
    ),

    # Binary service features (column names without '_Yes')
    'TechSupport': (
        'no tech support increases churn risk',
        'tech support subscription reduces churn risk',
    ),
    'OnlineSecurity': (
        'no online security increases churn risk',
        'online security subscription reduces churn risk',
    ),
    'OnlineBackup': (
        'no online backup increases churn risk',
        'online backup subscription reduces churn risk',
    ),
    'DeviceProtection': (
        'no device protection increases churn risk',
        'device protection subscription reduces churn risk',
    ),
    'StreamingTV': (
        'streaming TV subscription correlates with higher churn',
        'no streaming TV correlates with lower churn',
    ),
    'StreamingMovies': (
        'streaming movies subscription correlates with higher churn',
        'no streaming movies correlates with lower churn',
    ),
    'PaperlessBilling': (
        'paperless billing correlates with higher churn risk',
        'non-paperless billing correlates with lower churn risk',
    ),
    'Partner': (
        'absence of a partner correlates with higher churn',
        'having a partner correlates with lower churn',
    ),
    'Dependents': (
        'absence of dependents correlates with higher churn',
        'having dependents correlates with lower churn',
    ),
    'PhoneService': (
        'phone service subscription correlates with higher churn in this segment',
        'phone service subscription reduces churn risk in this segment',
    ),
    'MultipleLines': (
        'multiple phone lines correlate with higher churn',
        'single phone line correlates with lower churn',
    ),
    'gender': (
        'male gender correlates marginally with higher churn',
        'female gender correlates marginally with lower churn',
    ),
}


def shap_values_to_factors(
    shap_vals: list,
    feature_names: list,
    top_n: int = 3,
) -> tuple:
    """
    Translate SHAP values for one prediction into plain-English factors.

    Args:
        shap_vals: 1D array of SHAP values for a single prediction.
        feature_names: feature column names in same order as shap_vals.
        top_n: number of factors to return in each list.

    Returns:
        (risk_factors, protective_factors) — each a list of plain-English
        strings sorted by SHAP magnitude descending.
        risk_factors: features with shap > 0 (push toward churn).
        protective_factors: features with shap < 0 (push away from churn).
        Skips features where abs(shap) < 0.01 (noise floor).
        Fallback for missing templates: "{label} increases/reduces churn risk".
    """
    risk_factors = []
    protective_factors = []

    for shap_val, feature in zip(shap_vals, feature_names):
        if abs(shap_val) < 0.01:
            continue

        label = FEATURE_LABELS.get(feature, feature)
        templates = DIRECTION_TEMPLATES.get(feature)

        if shap_val > 0:
            text = templates[0] if templates else f'{label} increases churn risk'
            risk_factors.append((abs(shap_val), text))
        else:
            text = templates[1] if templates else f'{label} reduces churn risk'
            protective_factors.append((abs(shap_val), text))

    risk_factors.sort(key=lambda x: x[0], reverse=True)
    protective_factors.sort(key=lambda x: x[0], reverse=True)

    return (
        [text for _, text in risk_factors[:top_n]],
        [text for _, text in protective_factors[:top_n]],
    )
