from __future__ import annotations

import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.preprocess import prepare_data


def build_model() -> Pipeline:
    numeric_features = [
        "no_of_dependents",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
        "loan_to_income",
        "total_assets",
        "asset_to_income",
        "asset_to_loan",
    ]

    categorical_features = ["education", "self_employed"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    return model


def train_and_evaluate(data_path: str | Path | None = None) -> tuple[Pipeline, float]:
    X_train, X_test, X_all, y_train, y_test = prepare_data(data_path)
    model = build_model()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy


if __name__ == "__main__":
    model, acc = train_and_evaluate()
    print(f"Test accuracy: {acc:.4f}")
