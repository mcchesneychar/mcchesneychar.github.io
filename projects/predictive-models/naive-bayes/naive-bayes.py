"""
Naive Bayes — full-joint Bayes vs Naive Bayes (Laplace + Gaussian)
===================================================================
Charlie McChesney  ·  ECON 475 Section 4

A from-scratch implementation of:
  - The full-joint conditional probability and Bayesian MAP prediction.
  - The Naive Bayes assumption (feature independence) with Laplace
    smoothing for discrete features and a Gaussian likelihood for
    continuous ones.

Run:
    pip install -r requirements.txt
    python naive-bayes.py
"""

import os

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for the CSV alongside the script first (the layout inside the .zip
# download), then fall back to a data/ subfolder (the layout used on the live
# site so the in-browser demo can fetch it via data/...csv).
_CSV = "section4-dataset.csv"
DATA_PATH = next(
    p for p in (os.path.join(SCRIPT_DIR, _CSV), os.path.join(SCRIPT_DIR, "data", _CSV))
    if os.path.exists(p)
)


# =============================================================================
# A. Full-joint conditional probability  P(obsDF | obsT)
# =============================================================================

def cond_prob(df, t, obsDF, obsT):
    t_col = t.columns[0]
    t_val = obsT.iloc[0, 0]
    combined = pd.concat([df, t], axis=1)
    subset = combined[combined[t_col] == t_val]
    if len(subset) == 0:
        return 0.0
    obs_row = obsDF.iloc[0]
    mask = pd.Series([True] * len(subset), index=subset.index)
    for col in obsDF.columns:
        mask = mask & (subset[col] == obs_row[col])
    return len(subset[mask]) / len(subset)


# =============================================================================
# B. Bayesian MAP using full-joint conditional probability
# =============================================================================

def bayes(df, t, obsDF):
    t_col = t.columns[0]
    target_vals = t[t_col].unique()
    n_total = len(df)
    predictions = []
    for _, obs_row in obsDF.iterrows():
        obs_single = obs_row.to_frame().T.reset_index(drop=True)
        best_val, best_score = target_vals[0], -1.0
        for t_val in target_vals:
            obsT_single = pd.DataFrame({t_col: [t_val]})
            score = cond_prob(df, t, obs_single, obsT_single) * (t[t_col] == t_val).sum() / n_total
            if score > best_score:
                best_val, best_score = t_val, score
        predictions.append(best_val)
    return pd.Series(predictions, name="Predicted " + t_col)


# =============================================================================
# C & D. Naive Bayes with Laplace smoothing (discrete) or Gaussian (continuous)
# =============================================================================

def Ncond_prob(df, t, obsDF, obsT, k=1, feature="discrete_desc"):
    t_col = t.columns[0]
    t_val = obsT.iloc[0, 0]
    combined = pd.concat([df, t], axis=1)
    subset = combined[combined[t_col] == t_val]
    if len(subset) == 0:
        return 0.0
    obs_row = obsDF.iloc[0]
    product = 1.0

    if feature == "discrete_desc":
        for col in obsDF.columns:
            count_match = (subset[col] == obs_row[col]).sum()
            count_total = len(subset)
            domain_size = df[col].nunique()
            product *= (count_match + k) / (count_total + domain_size * k)

    elif feature == "continuous_desc":
        for col in obsDF.columns:
            mu = subset[col].mean()
            sigma = subset[col].std(ddof=0)
            x = obs_row[col]
            if sigma == 0:
                product *= 1.0 if x == mu else 0.0
            else:
                product *= (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    return product


def Nbayes(df, t, obsDF, k=1, feature="discrete_desc"):
    t_col = t.columns[0]
    target_vals = t[t_col].unique()
    n_total = len(df)
    predictions = []
    for _, obs_row in obsDF.iterrows():
        obs_single = obs_row.to_frame().T.reset_index(drop=True)
        best_val, best_score = target_vals[0], -1.0
        for t_val in target_vals:
            obsT_single = pd.DataFrame({t_col: [t_val]})
            score = Ncond_prob(df, t, obs_single, obsT_single, k=k, feature=feature) \
                    * (t[t_col] == t_val).sum() / n_total
            if score > best_score:
                best_val, best_score = t_val, score
        predictions.append(best_val)
    return pd.Series(predictions, name="Predicted " + t_col)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    df_raw = pd.read_csv(DATA_PATH)

    disc_cols = ["Married", "Urban", "Union", "HS Graduate",
                 "Brother Criminal", "Probation Before", "Charged Before"]
    cont_cols = ["Hours Worked", "Wage", "Education", "Work Experience"]
    t_col = "Committed Crime"

    df_disc = df_raw[disc_cols]
    df_cont = df_raw[cont_cols]
    t = df_raw[[t_col]]
    actual = t[t_col].reset_index(drop=True)

    print("=" * 60)
    print("Test 1 · Data loaded")
    print("=" * 60)
    print(f"Rows: {len(df_raw)}   Discrete features: {len(disc_cols)}   Continuous: {len(cont_cols)}")

    print("\n" + "=" * 60)
    print("Test 2 · bayes() — single observation")
    print("=" * 60)
    test_obs = df_disc.iloc[[0]]
    print("Features:", test_obs.to_dict(orient="records")[0])
    print("Predicted:", bayes(df_disc, t, test_obs).iloc[0])
    print("Actual:   ", t.iloc[0, 0])

    print("\n" + "=" * 60)
    print("Test 3 · Accuracy on discrete features (full dataset)")
    print("=" * 60)
    print("(Full-joint bayes() is slow — running...)")
    bayes_acc = (bayes(df_disc, t, df_disc).reset_index(drop=True) == actual).mean()
    print(f"bayes()   accuracy: {bayes_acc:.4f}")
    nbayes_acc = (Nbayes(df_disc, t, df_disc).reset_index(drop=True) == actual).mean()
    print(f"Nbayes()  accuracy: {nbayes_acc:.4f}")

    print("\n" + "=" * 60)
    print("Test 4 · Nbayes() — continuous features (Gaussian)")
    print("=" * 60)
    test_obs_cont = df_cont.iloc[[0]]
    print("Features:", test_obs_cont.to_dict(orient="records")[0])
    print("Predicted:", Nbayes(df_cont, t, test_obs_cont, feature="continuous_desc").iloc[0])
    print("Actual:   ", t.iloc[0, 0])
    cont_acc = (Nbayes(df_cont, t, df_cont, feature="continuous_desc").reset_index(drop=True) == actual).mean()
    print(f"Nbayes() continuous accuracy: {cont_acc:.4f}")
