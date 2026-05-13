"""
Nearest Neighbors — 1-NN, k-NN, and metric comparisons
=======================================================
Charlie McChesney  ·  ECON 475 Section 3

A from-scratch implementation of distance-based prediction (Euclidean,
Manhattan, Minkowski-3) plus a benchmark of sklearn's KNeighborsClassifier
across the kd_tree and brute algorithms, and a synthetic dataset where
cosine similarity meaningfully outperforms Minkowski.

Run:
    pip install -r requirements.txt
    python nearest-neighbors.py
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for the CSV alongside the script first (the layout inside the .zip
# download), then fall back to a data/ subfolder (the layout used on the live
# site so the in-browser demo can fetch it via data/...csv).
_CSV = "section3-dataset.csv"
DATA_PATH = next(
    p for p in (os.path.join(SCRIPT_DIR, _CSV), os.path.join(SCRIPT_DIR, "data", _CSV))
    if os.path.exists(p)
)


# =============================================================================
# A. Distance metrics + 1-NN prediction
# =============================================================================

def dist(a, b, method="Euclidean"):
    """Distance between two pandas Series."""
    if method == "Euclidean":
        return np.sqrt(((a - b) ** 2).sum())
    if method == "Manhattan":
        return (a - b).abs().sum()
    if method == "Minkowski3":
        return ((a - b).abs() ** 3).sum() ** (1 / 3)
    raise ValueError(f"Unknown method: {method}")


def predict(ds_train, t_train, ds_test):
    """1-NN prediction with Euclidean distance."""
    t_train = t_train.reset_index(drop=True)
    ds_train = ds_train.reset_index(drop=True)
    predictions = []
    for _, test_row in ds_test.iterrows():
        distances = ds_train.apply(lambda r: dist(test_row, r, method="Euclidean"), axis=1)
        predictions.append(t_train.iloc[distances.idxmin()])
    return predictions


# =============================================================================
# B. k-NN with plurality vote (Manhattan)
# =============================================================================

def predict2(ds_train, t_train, ds_test, k=5):
    """k-NN with plurality vote, Manhattan distance."""
    t_train = t_train.reset_index(drop=True)
    ds_train = ds_train.reset_index(drop=True)
    predictions = []
    for _, test_row in ds_test.iterrows():
        distances = ds_train.apply(lambda r: dist(test_row, r, method="Manhattan"), axis=1)
        k_nearest_idx = distances.nsmallest(k).index
        vote = t_train.iloc[k_nearest_idx].value_counts().idxmax()
        predictions.append(vote)
    return predictions


# =============================================================================
# C. KNeighborsClassifier — kd_tree vs brute timing
# =============================================================================

def KNN_Speed(ds_train, t_train, ds_test, t_test):
    """Returns [time_kd_tree, time_brute, acc_kd_tree, acc_brute]."""
    start = time.time()
    kd = KNeighborsClassifier(algorithm="kd_tree").fit(ds_train, t_train)
    kd_preds = kd.predict(ds_test)
    time_kd = time.time() - start

    start = time.time()
    bf = KNeighborsClassifier(algorithm="brute").fit(ds_train, t_train)
    bf_preds = bf.predict(ds_test)
    time_bf = time.time() - start

    return [time_kd, time_bf, accuracy_score(t_test, kd_preds), accuracy_score(t_test, bf_preds)]


# =============================================================================
# D. Synthetic dataset where cosine beats Minkowski
# =============================================================================

def dataCreation(seed=1):
    """Generate a synthetic dataset where cosine outperforms Euclidean.

    Class identity lives in the *direction* of feature vectors, not magnitude.
    Random magnitudes (1–1000) dominate Euclidean distance, but cosine —
    being magnitude-invariant — still cleanly separates the classes.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features, n_classes = 600, 10, 3
    base_dirs = np.eye(n_classes, n_features)

    X_list, y_list = [], []
    samples_per_class = n_samples // n_classes
    for cls in range(n_classes):
        noise = rng.normal(0, 0.15, size=(samples_per_class, n_features))
        directions = base_dirs[cls] + noise
        magnitudes = rng.uniform(1, 1000, size=(samples_per_class, 1))
        X_list.append(directions * magnitudes)
        y_list.extend([cls] * samples_per_class)

    X = np.vstack(X_list)
    y = np.array(y_list)
    ds_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    t_df = pd.DataFrame(y, columns=["target"])
    return t_df, ds_df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    target_col = "Committed Crime"
    feature_cols = [c for c in df.columns if c != target_col]
    X, y = df[feature_cols], df[target_col]

    # --- A. 1-NN, Euclidean (small test split) ----------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.05, random_state=42)
    print("=" * 60)
    print("A. 1-NN  (Euclidean, 5% test split)")
    print("=" * 60)
    preds = predict(X_tr, y_tr, X_te)
    print(f"Test accuracy: {accuracy_score(y_te, preds):.4f}")

    # --- B. 5-NN, Manhattan -----------------------------------------------
    print("\n" + "=" * 60)
    print("B. 5-NN  (Manhattan, plurality vote, 5% test split)")
    print("=" * 60)
    preds = predict2(X_tr, y_tr, X_te, k=5)
    print(f"Test accuracy: {accuracy_score(y_te, preds):.4f}")

    # --- C. KNN speed comparison ------------------------------------------
    print("\n" + "=" * 60)
    print("C. KNeighborsClassifier  (kd_tree vs brute, 80/20 split)")
    print("=" * 60)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.2, random_state=42)
    t_kd, t_bf, a_kd, a_bf = KNN_Speed(X_tr2, y_tr2, X_te2, y_te2)
    print(f"kd_tree   time={t_kd:.4f}s   acc={a_kd:.4f}")
    print(f"brute     time={t_bf:.4f}s   acc={a_bf:.4f}")

    # --- D. Synthetic data: cosine vs Minkowski ---------------------------
    print("\n" + "=" * 60)
    print("D. Synthetic dataset — cosine vs Minkowski(p=2)")
    print("=" * 60)
    t_df, ds_df = dataCreation(seed=1)
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(ds_df, t_df["target"], test_size=0.3, random_state=42)
    cos = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(Xs_tr, ys_tr)
    mnk = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2).fit(Xs_tr, ys_tr)
    print(f"cosine     acc={accuracy_score(ys_te, cos.predict(Xs_te)):.4f}")
    print(f"minkowski  acc={accuracy_score(ys_te, mnk.predict(Xs_te)):.4f}")
