"""
Regression with Gradient Descent — linear, logistic, + standardized
====================================================================
Charlie McChesney  ·  ECON 475 Section 5

OLS via statsmodels for reference, a Monte-Carlo sanity check on
parameter recovery, then from-scratch batch gradient descent for both
linear and logistic regression. The "extra credit" path uses feature
standardization so descent converges from a random init in <1000 steps.

Run:
    pip install -r requirements.txt
    python gradient-descent.py
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for the CSV alongside the script first (the layout inside the .zip
# download), then fall back to a data/ subfolder (the layout used on the live
# site so the in-browser demo can fetch it via data/...csv).
_CSV = "section5-dataset.csv"
DATA_PATH = next(
    p for p in (os.path.join(SCRIPT_DIR, _CSV), os.path.join(SCRIPT_DIR, "data", _CSV))
    if os.path.exists(p)
)


# =============================================================================
# A. OLS regression  (Wage ~ Education)
# =============================================================================

def ols_fit(df, target, feature):
    X = sm.add_constant(df[feature])
    return sm.OLS(df[target], X).fit()


# =============================================================================
# B. Monte Carlo recovery test  (true w0=2, w1=0.5)
# =============================================================================

def monte_carlo_recovery(true_w0=2.0, true_w1=0.5, n_obs=200, n_sims=1000, seed=42):
    rng = np.random.default_rng(seed)
    w0_est, w1_est = [], []
    for _ in range(n_sims):
        x = rng.uniform(0, 20, n_obs)
        y = true_w0 + true_w1 * x + rng.normal(0, 1, n_obs)
        res = sm.OLS(y, sm.add_constant(x)).fit()
        # Use .iloc for Series, raw indexing for ndarray
        params = res.params
        w0_est.append(params.iloc[0] if hasattr(params, "iloc") else params[0])
        w1_est.append(params.iloc[1] if hasattr(params, "iloc") else params[1])
    return float(np.mean(w0_est)), float(np.mean(w1_est))


# =============================================================================
# C. Linear gradient descent
# =============================================================================

def gradient_descent(ds, t, alpha, tol, max_steps, init_ws=(0.0, 0.0)):
    """Batch gradient descent for OLS-style linear regression."""
    x = np.asarray(ds, dtype=float)
    y = np.asarray(t, dtype=float)
    n = len(y)
    w0, w1 = float(init_ws[0]), float(init_ws[1])
    prev_loss = float("inf")
    for step in range(max_steps):
        err = (w0 + w1 * x) - y
        w0 -= alpha * (2.0 / n) * np.sum(err)
        w1 -= alpha * (2.0 / n) * np.sum(err * x)
        loss = np.mean(err ** 2)
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    return w0, w1, step + 1


# =============================================================================
# D. Logistic gradient descent
# =============================================================================

def _stable_sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def logistic_gradient_descent(ds, t, alpha, tol, max_steps, init_ws=(0.0, 0.0)):
    x = np.asarray(ds, dtype=float)
    y = np.asarray(t, dtype=float)
    n = len(y)
    w0, w1 = float(init_ws[0]), float(init_ws[1])
    prev_loss = float("inf")
    for step in range(max_steps):
        p = _stable_sigmoid(w0 + w1 * x)
        err = p - y
        w0 -= alpha * (1.0 / n) * np.sum(err)
        w1 -= alpha * (1.0 / n) * np.sum(err * x)
        eps = 1e-15
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        if loss > prev_loss:
            return w0, w1, step + 1   # safety brake: loss bumped up, halt
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    return w0, w1, step + 1


# =============================================================================
# Extra credit — standardized GD that converges in <1000 steps from random init
# =============================================================================

def gd_standardized(ds, t, alpha, tol, max_steps, init_ws, mode="linear"):
    """Standardize the feature so descent converges fast, then unstandardize."""
    x = np.asarray(ds, dtype=float)
    y = np.asarray(t, dtype=float)
    mu, sigma = x.mean(), x.std()
    x_std = (x - mu) / sigma
    n = len(y)
    w0, w1 = float(init_ws[0]), float(init_ws[1])
    prev_loss = float("inf")
    for step in range(max_steps):
        z = w0 + w1 * x_std
        if mode == "logistic":
            yhat = _stable_sigmoid(z)
        else:
            yhat = z
        err = yhat - y
        scale = 2.0 if mode == "linear" else 1.0
        w0 -= alpha * scale * (1.0 / n) * np.sum(err)
        w1 -= alpha * scale * (1.0 / n) * np.sum(err * x_std)
        if mode == "logistic":
            eps = 1e-15
            loss = -np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))
        else:
            loss = np.mean(err ** 2)
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    # un-standardize:  w0 + w1 * (x - mu) / sigma  ==  (w0 - w1*mu/sigma) + (w1/sigma) * x
    return w0 - w1 * mu / sigma, w1 / sigma, step + 1


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("A. OLS  (Wage ~ Education)")
    print("=" * 60)
    res = ols_fit(df, "Wage", "Education")
    print(f"w[0]={res.params.iloc[0]:.4f}  w[1]={res.params.iloc[1]:.4f}  R²={res.rsquared:.4f}")

    print("\n" + "=" * 60)
    print("B. Monte-Carlo OLS recovery (true w0=2.0, w1=0.5)")
    print("=" * 60)
    w0_mc, w1_mc = monte_carlo_recovery()
    print(f"mean w[0]={w0_mc:.4f}   mean w[1]={w1_mc:.4f}")

    print("\n" + "=" * 60)
    print("C. Linear gradient descent  (Committed Crime ~ Education)")
    print("=" * 60)
    ds = df["Education"].values
    t = df["Committed Crime"].values
    w0, w1, steps = gradient_descent(ds, t, alpha=0.005, tol=1e-9, max_steps=200000)
    print(f"w[0]={w0:.6f}   w[1]={w1:.6f}   steps={steps}")

    print("\n" + "=" * 60)
    print("D. Logistic gradient descent")
    print("=" * 60)
    w0, w1, steps = logistic_gradient_descent(ds, t, alpha=0.05, tol=1e-9, max_steps=200000)
    p_final = _stable_sigmoid(w0 + w1 * ds)
    print(f"w[0]={w0:.6f}   w[1]={w1:.6f}   steps={steps}")
    print(f"RMSE={mean_squared_error(t, p_final) ** 0.5:.6f}")

    print("\n" + "=" * 60)
    print("Extra credit · Standardized GD  (random init, <1000 steps)")
    print("=" * 60)
    rng = np.random.default_rng(0)
    print("Linear  (Committed Crime ~ Education):")
    for alpha in [0.01, 0.05, 0.1, 0.3, 0.5]:
        init = rng.uniform(-1, 1, size=2).tolist()
        w0, w1, steps = gd_standardized(ds, t, alpha=alpha, tol=1e-10, max_steps=1000, init_ws=init, mode="linear")
        print(f"  alpha={alpha:<5}  steps={steps:<5}  w0={w0:.4f}  w1={w1:.4f}")

    print("Logistic  (Committed Crime ~ Education):")
    for alpha in [0.05, 0.1, 0.3, 0.5, 1.0]:
        init = rng.uniform(-1, 1, size=2).tolist()
        w0, w1, steps = gd_standardized(ds, t, alpha=alpha, tol=1e-10, max_steps=1000, init_ws=init, mode="logistic")
        rmse = mean_squared_error(t, _stable_sigmoid(w0 + w1 * ds)) ** 0.5
        print(f"  alpha={alpha:<5}  steps={steps:<5}  w0={w0:.4f}  w1={w1:.4f}  RMSE={rmse:.4f}")
