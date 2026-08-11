from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "loan_approval_dataset.csv"


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load and lightly clean the raw dataset."""
    data_path = Path(path) if path is not None else DATA_PATH
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # Normalize target values and basic string cleanup
    df["loan_status"] = df["loan_status"].astype(str).str.strip()
    df["education"] = df["education"].astype(str).str.strip()
    df["self_employed"] = df["self_employed"].astype(str).str.strip()

    # Basic invalid-value handling that is safe before train/test split
    for col in [
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0)
        df[col] = df[col].clip(lower=0)

    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})
    return df


def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix and target vector."""
    df = df.copy()

    # Derived financial features
    df["loan_to_income"] = df["loan_amount"] / (df["income_annum"] + 1)
    df["total_assets"] = (
        df["residential_assets_value"]
        + df["commercial_assets_value"]
        + df["luxury_assets_value"]
        + df["bank_asset_value"]
    )
    df["asset_to_income"] = df["total_assets"] / (df["income_annum"] + 1)
    df["asset_to_loan"] = df["total_assets"] / (df["loan_amount"] + 1)

    # Keep the identifier out of modeling
    feature_frame = df.drop(columns=["loan_id", "loan_status"], errors="ignore")
    target = df["loan_status"]
    return feature_frame, target


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create train/test splits with stratification."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessing_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Convenience wrapper for loading and creating model-ready features."""
    cleaned_df = load_data(df if isinstance(df, pd.DataFrame) else None)
    X, y = make_features(cleaned_df)
    return X, y


def prepare_data(path: str | Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, engineer features, and split into train/test sets."""
    df = load_data(path)
    X, y = make_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, X, y_train, y_test
