import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

def run_predictive_model(
    data: pd.DataFrame,
    target: str,
    model_type: str = "auto",
    use_random_forest: bool = True
):

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data.")

    df = data.copy().dropna(subset=[target])

    y = df[target]
    X = df.drop(columns=[target])

    # One-hot encoding (safe)
    X = pd.get_dummies(X, drop_first=True)

    # Auto-detect task
    if model_type == "auto":
        model_type = "classification" if y.nunique() <= 10 and y.dtype in ("int", "bool") else "regression"

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Choose model
    if model_type == "classification":
        if use_random_forest:
            model = RandomForestClassifier(
                random_state=42, n_estimators=200, max_depth=8
            )
        else:
            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced"   
            )
    else:
        if use_random_forest:
            model = RandomForestRegressor(
                random_state=42, n_estimators=300, max_depth=8
            )
        else:
            model = LinearRegression()

    # Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", model)
    ])

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    results = {"model_type": model_type}

    # ----------------------------
    # CLASSIFICATION
    # ----------------------------
    if model_type == "classification":
        probs = pipeline.predict_proba(X_test)[:, 1]

        # threshold tuning
        preds = (probs > 0.2).astype(int)

        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, probs) if len(set(y_test)) == 2 else None

        results.update({
            "accuracy": acc,
            "roc_auc": roc
        })

    # ----------------------------
    # REGRESSION
    # ----------------------------
    else:
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.update({
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

    print(f"✅ Trained {model_type} model using {model.__class__.__name__}")
    print("Results:", results)

    y_pred = pipeline.predict(X_test)

    return pipeline, results, X_test, y_test