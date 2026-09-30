"""
gradient_descent_from_scratch.py
---------------------------------
Linear regression using ONLY numpy — no sklearn — so you can see exactly
what .fit() was hiding from you. Trains on the same housing.csv.

Run: python -m experiments.scratch_gradient_descent
"""

import numpy as np
import pandas as pd

df = pd.read_csv("data/housing.csv").dropna()

# Use just the numeric features for simplicity (no one-hot encoding logic
# to write by hand right now — the point here is the training loop, not
# preprocessing, which you already understand from train.py).
features = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km"]
X = df[features].values.astype(float)
y = df["price"].values.astype(float)

# --- Standardize manually (mean 0, std 1) --------------------------------
# Without this, gradient descent behaves badly: area_sqft (~1500) and
# bedrooms (~3) are on wildly different scales, so a single learning rate
# can't suit both at once.
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_scaled = (X - X_mean) / X_std

n_samples, n_features = X_scaled.shape

# --- Initialize weights and bias -----------------------------------------
w = np.zeros(n_features)   # start with all weights at 0 -- a genuine "blank slate"
b = 0.0

learning_rate = 0.5
n_iterations = 500

# --- Training loop: this IS what .fit() does internally for models
#     that don't have a closed-form solution (e.g. logistic regression,
#     neural networks). Linear regression COULD skip this loop via the
#     normal equation, but we're doing it manually to see the mechanics. --
for iteration in range(n_iterations):
    y_pred = X_scaled @ w + b               # forward pass: current predictions

    error = y_pred - y
    mse = np.mean(error ** 2)                # current loss

    # Gradients: calculus tells us these are the partial derivatives
    # of MSE with respect to w and b.
    dw = (2 / n_samples) * (X_scaled.T @ error)
    db = (2 / n_samples) * np.sum(error)

    # The actual "learning" step: move weights opposite the gradient.
    w -= learning_rate * dw
    b -= learning_rate * db

    if iteration % 50 == 0 or iteration == n_iterations - 1:
        print(f"Iteration {iteration:4d} | MSE: {mse:,.0f} | RMSE: {np.sqrt(mse):,.0f}")

print("\nFinal learned weights (on standardized features):")
for name, weight in zip(features, w):
    print(f"  {name:22s}: {weight:10.2f}")
print(f"  {'bias (intercept)':22s}: {b:10.2f}")

print("\nCompare this RMSE to sklearn's ~16,130 from train.py's log — should be close,")
print("though not identical, since this uses only numeric features (no location_tier).")
