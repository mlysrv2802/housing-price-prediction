"""
eda_feature_analysis.py
------------------------
Run this BEFORE training, on any new dataset — this is the step most
tutorials skip, but real ML work always starts here.

What this script does, and why each step matters:
  1. Basic shape/missingness check       -> catches data quality issues early
  2. Correlation with target             -> quick signal on which features matter
  3. Multicollinearity check (VIF)       -> flags redundant features
  4. Feature importance via RandomForest -> a fast, model-based sanity check
     that often agrees with (or corrects) plain correlation

Run: python -m experiments.eda_feature_analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from housing import config
from housing.data.data_loader import load_raw_data


def _compute_vif(numeric_df: pd.DataFrame) -> pd.DataFrame:
    """Manual VIF computation (avoids a statsmodels dependency).

    For each feature, regress it on ALL other features. VIF = 1 / (1 - R^2)
    of that regression. High R^2 (the feature is well-predicted by the
    others) -> high VIF -> redundant feature.
    """
    results = []
    columns = numeric_df.columns
    for col in columns:
        y = numeric_df[col].values
        X = numeric_df.drop(columns=[col]).values
        r2 = LinearRegression().fit(X, y).score(X, y)
        vif = 1.0 / (1.0 - r2) if r2 < 0.999 else float("inf")
        results.append({"feature": col, "VIF": vif})
    return pd.DataFrame(results)


def basic_overview(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("1. BASIC OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("Missing values per column:")
    missing = df.isna().sum()
    missing_only = missing[missing > 0]
    print(missing_only if not missing_only.empty else "  (none)")
    print()
    print("Numeric feature summary:")
    print(df.describe().T[["mean", "std", "min", "max"]])
    print()


def correlation_with_target(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("2. CORRELATION WITH TARGET")
    print("=" * 60)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)
    corr = corr.sort_values(key=abs, ascending=False)
    print(corr)
    print("\nRule of thumb (not a strict cutoff): |corr| < 0.1 often means")
    print("the feature is contributing little on its own -- but check")
    print("feature importance below too, since correlation misses")
    print("interactions between features.\n")


def multicollinearity_check(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("3. MULTICOLLINEARITY CHECK (VIF)")
    print("=" * 60)
    print("VIF measures how much a feature can be predicted FROM the")
    print("other features. VIF > 5-10 signals redundancy -- two features")
    print("carrying overlapping information, which can distort a linear")
    print("model's individual coefficients (though it may not hurt raw")
    print("predictive accuracy much).\n")

    numeric_df = df[config.NUMERIC_FEATURES].dropna()
    vif_data = _compute_vif(numeric_df)
    print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))
    print()


def feature_importance_via_forest(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("4. FEATURE IMPORTANCE (RandomForest -- model-based ranking)")
    print("=" * 60)
    print("A quick, practical way to rank ALL features (numeric AND")
    print("categorical) together by how much they actually reduce")
    print("prediction error -- more reliable than raw correlation because")
    print("it captures non-linear effects and feature interactions.\n")

    df_clean = df.dropna()
    X_numeric = df_clean[config.NUMERIC_FEATURES].values

    encoder = OneHotEncoder(sparse_output=False)
    X_categorical = encoder.fit_transform(df_clean[config.CATEGORICAL_FEATURES])
    categorical_names = encoder.get_feature_names_out(config.CATEGORICAL_FEATURES)

    X = np.hstack([X_numeric, X_categorical])
    feature_names = config.NUMERIC_FEATURES + list(categorical_names)
    y = df_clean[config.TARGET_COLUMN].values

    forest = RandomForestRegressor(n_estimators=200, random_state=config.RANDOM_STATE)
    forest.fit(X, y)

    importance = pd.Series(forest.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)
    print(importance.to_string())
    print()


if __name__ == "__main__":
    df = load_raw_data()
    basic_overview(df)
    correlation_with_target(df)
    multicollinearity_check(df)
    feature_importance_via_forest(df)
