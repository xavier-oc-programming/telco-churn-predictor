"""
shap_to_language.py

Translates raw SHAP values for a single churn prediction into plain-English
factors that a non-technical stakeholder can understand and act on.

Imported by app.py at startup. No classes — module-level dicts and one function.
"""

# Maps every preprocessed feature name (including one-hot encoded columns) to
# a human-readable label shown in the UI and notebook.
FEATURE_LABELS = {
    # Numeric features
    'tenure': 'Customer tenure',
    'MonthlyCharges': 'Monthly charges',
    'TotalCharges': 'Total charges',
    'SeniorCitizen': 'Senior citizen',

    # Contract type (one-hot, drop_first removes Month-to-month as baseline)
    'Contract_One year': 'One-year contract',
    'Contract_Two year': 'Two-year contract',

    # Internet service (drop_first removes DSL as baseline)
    'InternetService_Fiber optic': 'Fiber optic internet',
    'InternetService_No': 'No internet service',

    # Payment method (drop_first removes Bank transfer as baseline)
    'PaymentMethod_Credit card (automatic)': 'Credit card (automatic) payment',
    'PaymentMethod_Electronic check': 'Electronic check payment',
    'PaymentMethod_Mailed check': 'Mailed check payment',

    # Binary Yes/No service features
    'TechSupport_Yes': 'Tech support subscription',
    'OnlineSecurity_Yes': 'Online security subscription',
    'OnlineBackup_Yes': 'Online backup subscription',
    'DeviceProtection_Yes': 'Device protection subscription',
    'StreamingTV_Yes': 'Streaming TV subscription',
    'StreamingMovies_Yes': 'Streaming movies subscription',

    # Paperless billing
    'PaperlessBilling_Yes': 'Paperless billing',

    # Partner / Dependents / Phone
    'Partner_Yes': 'Has a partner',
    'Dependents_Yes': 'Has dependents',
    'PhoneService_Yes': 'Has phone service',
    'MultipleLines_Yes': 'Multiple phone lines',

    # Gender (encoded but low SHAP signal in practice)
    'gender_Male': 'Gender: male',
}

# Maps each feature name to (text_when_shap_positive, text_when_shap_negative).
# shap > 0 → feature pushes the prediction toward churn.
# shap < 0 → feature pushes the prediction away from churn (protective).
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

    # Support and security services
    'TechSupport_Yes': (
        'no tech support increases churn risk',
        'tech support subscription reduces churn risk',
    ),
    'OnlineSecurity_Yes': (
        'no online security increases churn risk',
        'online security subscription reduces churn risk',
    ),
    'OnlineBackup_Yes': (
        'no online backup increases churn risk',
        'online backup subscription reduces churn risk',
    ),
    'DeviceProtection_Yes': (
        'no device protection increases churn risk',
        'device protection subscription reduces churn risk',
    ),
    'StreamingTV_Yes': (
        'streaming TV subscription correlates with higher churn',
        'no streaming TV correlates with lower churn',
    ),
    'StreamingMovies_Yes': (
        'streaming movies subscription correlates with higher churn',
        'no streaming movies correlates with lower churn',
    ),

    # Billing and demographics
    'PaperlessBilling_Yes': (
        'paperless billing correlates with higher churn risk',
        'non-paperless billing correlates with lower churn risk',
    ),
    'Partner_Yes': (
        'absence of a partner correlates with higher churn',
        'having a partner correlates with lower churn',
    ),
    'Dependents_Yes': (
        'absence of dependents correlates with higher churn',
        'having dependents correlates with lower churn',
    ),
    'PhoneService_Yes': (
        'phone service subscription correlates with higher churn in this segment',
        'no phone service correlates with lower churn in this segment',
    ),
    'MultipleLines_Yes': (
        'multiple phone lines correlate with higher churn',
        'single phone line correlates with lower churn',
    ),
    'gender_Male': (
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
            if templates:
                text = templates[0]
            else:
                text = f'{label} increases churn risk'
            risk_factors.append((abs(shap_val), text))
        else:
            if templates:
                text = templates[1]
            else:
                text = f'{label} reduces churn risk'
            protective_factors.append((abs(shap_val), text))

    # Sort each list by SHAP magnitude descending, return only the text strings
    risk_factors.sort(key=lambda x: x[0], reverse=True)
    protective_factors.sort(key=lambda x: x[0], reverse=True)

    return (
        [text for _, text in risk_factors[:top_n]],
        [text for _, text in protective_factors[:top_n]],
    )
