"""
Trains a RandomForestRegressor to predict mental_health_score from
social media usage + lifestyle features, evaluates it, and saves:
  - model.pkl            (trained pipeline: preprocessing + model)
  - feature_importance.csv
  - metrics.json
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from generate_data import generate_dataset

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "age",
    "daily_social_media_hours",
    "num_platforms_used",
    "sleep_hours",
    "physical_activity_hours",
    "screen_time_before_bed_min",
    "study_hours",
    "face_to_face_hours",
    "academic_performance",
    "social_comparison_score",
    "cyberbullying_experience",
]
CATEGORICAL_FEATURES = ["gender"]
TARGET = "mental_health_score"


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipe


def validate_dataframe(df):
    """Checks an uploaded dataframe has every required column with sane types.
    Returns (is_valid: bool, message: str)."""
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        return False, f"Missing required column(s): {', '.join(sorted(missing))}"

    if len(df) < 30:
        return False, "Need at least 30 rows to train a meaningful model."

    bad_genders = set(df["gender"].dropna().astype(str).unique()) - {"Male", "Female", "Other"}
    if bad_genders:
        return False, (
            f"'gender' column has unexpected values: {sorted(bad_genders)}. "
            "Expected only Male / Female / Other."
        )

    for col in NUMERIC_FEATURES + [TARGET]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must be numeric."

    if df[TARGET].min() < 0 or df[TARGET].max() > 100:
        return False, "'mental_health_score' should be on a 0-100 scale."

    return True, "Looks good!"


def train_and_save(df, model_path="model.pkl", data_path="student_mental_health_data.csv",
                    metrics_path="metrics.json", fi_path="feature_importance.csv"):
    """Trains a pipeline on `df`, saves model + dataset + metrics + feature
    importance to disk, and returns (pipe, metrics_dict, feature_importance_df)."""
    df = df.copy()
    df["cyberbullying_experience"] = df["cyberbullying_experience"].astype(int)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 3),
               "n_train": len(X_train), "n_test": len(X_test)}

    ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_names
    importances = pipe.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)

    if model_path:
        joblib.dump(pipe, model_path)
    if data_path:
        df.to_csv(data_path, index=False)
    if metrics_path:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    if fi_path:
        fi_df.to_csv(fi_path, index=False)

    return pipe, metrics, fi_df


def main():
    df = generate_dataset()
    pipe, metrics, fi_df = train_and_save(df)
    print(f"MAE:  {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2:   {metrics['R2']:.3f}")
    print("\nTop features:\n", fi_df.head(8))
    print("\nSaved model.pkl")


if __name__ == "__main__":
    main()
