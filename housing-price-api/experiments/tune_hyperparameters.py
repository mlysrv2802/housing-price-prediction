"""
tune_hyperparameters.py
------------------------
Systematically searches for the best alpha (regularization strength) for
Ridge and Lasso, using k-fold cross-validation -- instead of the manual
guess-and-check we just did in compare_models.py.

Run: python -m experiments.tune_hyperparameters
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
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


def tune(model_name: str, model, param_grid: dict, X_train, y_train, X_test, y_test):
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", model),
    ])

    # cv=5 -> 5-fold cross-validation on the TRAINING set only.
    # scoring uses negative MSE because GridSearchCV always maximizes a
    # score, and lower MSE is better, so we negate it to fit that convention.
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,   # use all CPU cores -- these fits are independent
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    test_rmse = np.sqrt(-search.score(X_test, y_test))
    test_r2 = best_pipeline.score(X_test, y_test)

    print(f"\n--- {model_name} ---")
    print(f"Best params (via 5-fold CV): {search.best_params_}")
    print(f"CV best RMSE (avg across folds): {np.sqrt(-search.best_score_):,.0f}")
    print(f"Held-out test RMSE: {test_rmse:,.0f}")
    print(f"Held-out test R^2:  {test_r2:.4f}")

    return search


def main() -> None:
    df = load_raw_data()
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # The search space -- values on a LOG scale, since regularization
    # strength effects are multiplicative, not additive. This is standard
    # practice: 1, 10, 100 matters more than 1, 2, 3.
    alpha_grid = [0.01, 0.1, 1, 10, 50, 100, 500, 1000, 5000]

    tune(
        "Ridge", Ridge(random_state=config.RANDOM_STATE),
        {"regressor__alpha": alpha_grid},
        X_train, y_train, X_test, y_test,
    )

    tune(
        "Lasso", Lasso(random_state=config.RANDOM_STATE, max_iter=10000),
        {"regressor__alpha": alpha_grid},
        X_train, y_train, X_test, y_test,
    )


if __name__ == "__main__":
    main()
