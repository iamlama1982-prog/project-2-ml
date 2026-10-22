from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the Telco churn data and apply the project cleaning rules.

    Returns
    -------
    pd.DataFrame
        Cleaned customer data with ``TotalCharges`` converted to numeric,
        new-customer blanks filled with 0, and ``customerID`` removed.
    """

    data = pd.read_csv(RAW / "telco_churn.csv")

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce",
    )

    # The 11 blanks belong to customers with tenure = 0, so zero is meaningful.
    data["TotalCharges"] = data["TotalCharges"].fillna(0)

    # customerID identifies a person but contains no predictive signal.
    data = data.drop(columns=["customerID"])

    return data


def load_and_split_data(test_size: float = 0.20):
    """Load the cleaned data and reproduce the fixed train/test split.

    Parameters
    ----------
    test_size : float, default=0.20
        Share of rows reserved for the untouched test set.

    Returns
    -------
    tuple
        ``X_train, X_test, y_train, y_test`` using ``RANDOM_STATE`` and a
        stratified split so the churn rate stays similar in both sets.
    """

    data = load_data()

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(numeric, categorical) -> ColumnTransformer:
    """Create the preprocessing used by every tabular model.

    Parameters
    ----------
    numeric : list-like
        Names of numeric columns. Missing values are median-imputed and values
        are standardised.
    categorical : list-like
        Names of categorical columns. Missing values are filled with the most
        frequent category and values are one-hot encoded.

    Returns
    -------
    ColumnTransformer
        A transformer that can be placed inside a scikit-learn ``Pipeline`` so
        preprocessing is fitted separately inside each cross-validation fold.
    """

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    return preprocessor


def compare_models(models, X, y, cv, scoring) -> pd.DataFrame:
    """Cross-validate several models on the same folds and metric.

    Parameters
    ----------
    models : dict
        Mapping of display name to estimator or pipeline.
    X : pandas DataFrame
        Training features.
    y : pandas Series
        Training target.
    cv : cross-validation splitter
        Shared folds used for every model.
    scoring : scorer
        scikit-learn scorer used for comparison.

    Returns
    -------
    pd.DataFrame
        Model name, mean recall and standard deviation, ordered best to worst.
    """

    results = []

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
        )

        results.append({
            "Model": name,
            "Mean Recall": scores.mean(),
            "Std Recall": scores.std(),
        })

    return (
        pd.DataFrame(results)
        .sort_values("Mean Recall", ascending=False)
        .reset_index(drop=True)
    )


def add_tenure_group(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``X`` with tenure grouped into four customer-life stages.

    The feature is deliberately simple and is tested inside the modelling
    pipeline. It is only retained if cross-validation shows an improvement.

    Parameters
    ----------
    X : pd.DataFrame
        Customer feature data containing a ``tenure`` column.

    Returns
    -------
    pd.DataFrame
        Copy of the input with a categorical ``TenureGroup`` column added.
    """

    X_with_group = X.copy()

    X_with_group["TenureGroup"] = pd.cut(
        X_with_group["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months",
        ],
    )

    return X_with_group
