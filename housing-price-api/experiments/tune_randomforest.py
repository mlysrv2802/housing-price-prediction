"""
tune_randomforest.py
----------------------
Demonstrates RandomizedSearchCV -- necessary once you have MULTIPLE
hyperparameters to tune together (unlike Ridge/Lasso's single `alpha`,
where GridSearchCV was fine).

RandomForest has several meaningful hyperparameters:
  - n_estimators:      how many trees in the forest
  - max_depth:         how deep each tree can grow (controls overfitting)
  - min_samples_split: min samples needed to split a node
  - min_samples_leaf:  min samples required at a leaf node

Trying every combination (GridSearchCV) of even modest ranges for these
four would mean thousands of model fits. RandomizedSearchCV samples a
fixed budget (n_iter) of random combinations instead -- much cheaper,
usually nearly as good.

Run: python -m experiments.tune_randomforest
"""

import numpy as np
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from housing import config
from housing.data.data_loader import load_raw_data


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, config.NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURES),
    ])


def main() -> None:
    df = load_raw_data()
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(random_state=config.RANDOM_STATE)),
    ])

    # Distributions/ranges to sample from, NOT a fixed exhaustive list.
    # randint(a, b) means "pick a random integer between a and b each trial"
    # -- this is the key difference from GridSearchCV's fixed value lists.
    param_distributions = {
        "regressor__n_estimators": randint(50, 500),
        "regressor__max_depth": randint(3, 30),
        "regressor__min_samples_split": randint(2, 20),
        "regressor__min_samples_leaf": randint(1, 10),
    }

    # n_iter=30 -> only 30 random combinations get tried, out of what would
    # be thousands if we did a full grid over these same ranges.
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    test_rmse = np.sqrt(-search.score(X_test, y_test))
    test_r2 = best_pipeline.score(X_test, y_test)

    print(f"Best params (from 30 random samples): {search.best_params_}")
    print(f"CV best RMSE: {np.sqrt(-search.best_score_):,.0f}")
    print(f"Held-out test RMSE: {test_rmse:,.0f}")
    print(f"Held-out test R^2:  {test_r2:.4f}")
    print()
    print("Compare this R^2 to compare_models.py's plain RandomForest (0.7547)")
    print("-- tuning should meaningfully close some of the gap to Ridge (0.8206),")
    print("though tree models may still not beat linear ones on this genuinely")
    print("linear-ish dataset. That itself is a valid, useful finding.")


if __name__ == "__main__":
    main()
