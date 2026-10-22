"""Reusable scoring helpers for the Streamlit churn demo.

These functions contain the model-loading and business-ranking logic without any
Streamlit UI code, which keeps the demo testable and makes the operational logic
clearer to read.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.functions import RANDOM_STATE, build_preprocessor


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
MODEL_PATH = ROOT / "models" / "telco_churn_model.joblib"

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

MODEL_THRESHOLD = 0.50

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

MODEL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def load_raw_data() -> pd.DataFrame:
    """Load IBM Telco data locally, or from IBM if the raw file is absent."""

    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)

    return pd.read_csv(TELCO_URL)


def clean_scoring_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready feature columns from raw or uploaded customer data."""

    scoring_data = data.copy()

    scoring_data["TotalCharges"] = pd.to_numeric(
        scoring_data["TotalCharges"],
        errors="coerce",
    )
    scoring_data["TotalCharges"] = scoring_data["TotalCharges"].fillna(0)

    columns_to_drop = []
    if "customerID" in scoring_data.columns:
        columns_to_drop.append("customerID")
    if "Churn" in scoring_data.columns:
        columns_to_drop.append("Churn")

    if columns_to_drop:
        scoring_data = scoring_data.drop(columns=columns_to_drop)

    missing_columns = []
    for column in MODEL_FEATURES:
        if column not in scoring_data.columns:
            missing_columns.append(column)

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            "The customer file is missing required columns: " + missing_text
        )

    return scoring_data[MODEL_FEATURES]


def create_final_model() -> Pipeline:
    """Create the exact final pipeline selected in the project notebooks."""

    categorical_features = []
    for column in MODEL_FEATURES:
        if column not in NUMERIC_FEATURES:
            categorical_features.append(column)

    preprocessor = build_preprocessor(
        NUMERIC_FEATURES,
        categorical_features,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            C=1,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])

    return model


def load_model_and_demo_split():
    """Load the saved model when possible and prepare the held-out demo data.

    Returns
    -------
    tuple
        Fitted model, threshold, held-out X/y, held-out customer IDs and a short
        description of whether the saved artefact or rebuilt final pipeline was
        used.
    """

    raw_data = load_raw_data()
    model_data = clean_scoring_frame(raw_data)
    target = raw_data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        model_data,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model_source = "Saved fitted pipeline"
    threshold = MODEL_THRESHOLD
    model = None

    if MODEL_PATH.exists():
        try:
            saved = joblib.load(MODEL_PATH)
            model = saved["model"]
            threshold = float(saved["threshold"])
        except Exception:
            # joblib artefacts are version-sensitive. Rebuilding from the final
            # documented specification is safer than allowing a demo failure.
            model = None

    if model is None:
        model = create_final_model()
        model.fit(X_train, y_train)
        model_source = "Final pipeline rebuilt from project specification"

    customer_ids = pd.Series(
        raw_data.loc[X_test.index, "customerID"].values,
        index=X_test.index,
        name="customerID",
    )

    return model, threshold, X_test, y_test, customer_ids, model_source


def churn_scores(model, customer_data: pd.DataFrame) -> np.ndarray:
    """Return the model score for the positive class, ``Churn = Yes``."""

    probabilities = model.predict_proba(customer_data)
    classes = list(model.named_steps["model"].classes_)
    yes_index = classes.index("Yes")
    return probabilities[:, yes_index]


def priority_band(score: float, threshold: float) -> str:
    """Translate a model score into a simple operational priority band."""

    if score >= 0.75:
        return "Priority 1"
    if score >= threshold:
        return "Priority 2"
    if score >= 0.35:
        return "Watch"
    return "Routine"


def action_for_score(score: float, threshold: float) -> str:
    """Suggest a management next step without automating a commercial action."""

    if score >= 0.75:
        return (
            "Contact early. Check recent service history, complaints and the current "
            "account before deciding what, if anything, is worth offering."
        )
    if score >= threshold:
        return (
            "Put this customer in the next retention queue. Check the current account context "
            "before contact — the model is telling you who to look at, not what deal to give them."
        )
    if score >= 0.35:
        return (
            "Keep on the watchlist. No proactive contact is needed unless new service or "
            "complaint information gives you another reason to act."
        )
    return (
        "Routine service. The model does not currently prioritise this customer "
        "for retention outreach."
    )


def add_scores(model, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Score a customer dataframe and add ranking fields."""

    scoring_data = clean_scoring_frame(data)
    scores = churn_scores(model, scoring_data)

    result = data.copy()
    result["Risk score"] = scores

    priorities = []
    for score in scores:
        priorities.append(priority_band(float(score), threshold))
    result["Priority"] = priorities

    return result.sort_values("Risk score", ascending=False)


def friendly_feature_name(raw_name: str) -> str:
    """Turn a transformed feature name into readable demo copy."""

    name = raw_name
    name = name.replace("numeric__", "")
    name = name.replace("categorical__", "")
    name = name.replace("_", " = ", 1)

    replacements = {
        "MonthlyCharges": "Monthly charges",
        "TotalCharges": "Total charges",
        "tenure": "Tenure",
        "Fiber optic": "Fibre optic",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    return name


def customer_drivers(model, customer_row: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return the strongest row-level pushes towards churn and staying."""

    transformed = model.named_steps["preprocessor"].transform(customer_row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    contributions = transformed[0] * coefficients

    contribution_table = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions,
    })

    positive = contribution_table[
        contribution_table["Contribution"] > 0
    ].sort_values("Contribution", ascending=False)

    negative = contribution_table[
        contribution_table["Contribution"] < 0
    ].sort_values("Contribution", ascending=True)

    positive_names = []
    for feature in positive.head(3)["Feature"]:
        positive_names.append(friendly_feature_name(feature))

    negative_names = []
    for feature in negative.head(3)["Feature"]:
        negative_names.append(friendly_feature_name(feature))

    return positive_names, negative_names


def scenario_customers() -> dict:
    """Return the three presentation customers used for the live demo."""

    return {
        "High-risk example": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.00,
            "TotalCharges": 285.00,
        },
        "Low-risk example": {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 60.00,
            "TotalCharges": 3600.00,
        },
        "Borderline example": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 18,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 80.00,
            "TotalCharges": 1440.00,
        },
    }
