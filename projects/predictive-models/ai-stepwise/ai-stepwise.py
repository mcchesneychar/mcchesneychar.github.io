"""
AI-Augmented Stepwise Regression — ECON 475 Final Project
==========================================================
Charlie McChesney  ·  ECON 475 Generative AI Final Project

Three forward-stepwise regressions on the California Housing dataset,
compared by out-of-sample RMSE:

    Model A — Baseline stepwise on the 8 raw features.
    Model B — AI-augmented stepwise. The LLM is NOT called at runtime —
              the prompt and the JSON returned during the live demo are
              embedded below as a comment block, and the suggested
              features are generated in plain pandas. This keeps the
              script offline and dependency-free.
    Model C — Kitchen-sink stepwise on every pairwise product + square
              of the raw features (44 candidates total).

Run:
    pip install -r requirements.txt
    python ai-stepwise.py


# --- (Optional) Live LLM call to regenerate the feature suggestions ---------
# This script ships with a pre-baked Claude response (see the JSON block
# further down) so it runs offline with no API key. If you'd like to
# regenerate the feature suggestions with a live LLM call instead, the
# wiring would look roughly like this:
#
#   from dotenv import load_dotenv          # pip install python-dotenv
#   import os, json
#   load_dotenv()                           # reads .env from this directory
#   API_KEY = os.environ["LLM_API_KEY"]
#
#   # Then call your provider of choice (Anthropic / Gemini / OpenAI) with
#   # the PROMPT below, parse the JSON response, and pass the parsed list
#   # of {name, formula, rationale} dicts into a feature builder that
#   # mirrors add_ai_features() instead of using the hard-coded version.
#
# A sample .env would contain:
#   LLM_API_KEY=your_key_here
#
# Add `.env` to your .gitignore so the key never gets committed.
# ---------------------------------------------------------------------------
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# =============================================================================
# Setup
# =============================================================================
data = fetch_california_housing(as_frame=True)
X, y = data.data.copy(), data.target.copy()
RAW_COLS = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# =============================================================================
# Helpers
# =============================================================================

def forward_stepwise_aic(X_tr: pd.DataFrame, y_tr: pd.Series) -> list:
    """Forward stepwise selection minimizing AIC."""
    remaining, selected = list(X_tr.columns), []
    current_aic = sm.OLS(y_tr, sm.add_constant(pd.DataFrame(index=X_tr.index))).fit().aic
    improved = True
    while remaining and improved:
        improved = False
        best_aic, best_var = current_aic, None
        for var in remaining:
            model = sm.OLS(y_tr, sm.add_constant(X_tr[selected + [var]])).fit()
            if model.aic < best_aic:
                best_aic, best_var = model.aic, var
        if best_var is not None:
            selected.append(best_var)
            remaining.remove(best_var)
            current_aic = best_aic
            improved = True
    return selected


def fit_and_rmse(X_tr, y_tr, X_te, y_te, selected) -> float:
    model = sm.OLS(y_tr, sm.add_constant(X_tr[selected])).fit()
    preds = model.predict(sm.add_constant(X_te[selected], has_constant="add"))
    return float(np.sqrt(mean_squared_error(y_te, preds)))


def drop_bad_columns(df_tr, df_te):
    dropped = []
    for col in list(df_tr.columns):
        if (df_tr[col].isna().any() or df_te[col].isna().any()
                or np.isinf(df_tr[col]).any() or np.isinf(df_te[col]).any()):
            dropped.append(col)
            df_tr = df_tr.drop(columns=col)
            df_te = df_te.drop(columns=col)
    return df_tr, df_te, dropped


# =============================================================================
# Model A — Baseline
# =============================================================================
print("=" * 72)
print("MODEL A — Baseline stepwise (8 raw features)")
print("=" * 72)
selected_A = forward_stepwise_aic(X_train, y_train)
rmse_A = fit_and_rmse(X_train, y_train, X_test, y_test, selected_A)
candidates_A = len(RAW_COLS)
print(f"Candidate features: {candidates_A}")
print(f"Selected ({len(selected_A)}): {selected_A}")
print(f"Test RMSE: {rmse_A:.4f}")


# =============================================================================
# Model B — AI-augmented (pre-baked Claude response)
# =============================================================================
# --- Live-demo log: prompt + Claude's JSON response (claude-opus-4-7) -------
# Reproduced verbatim so anyone reading this file can see exactly what the
# model returned. The features built below match this JSON one-for-one.
#
# --- PROMPT -----------------------------------------------------------------
# You are helping a predictive-analytics student improve a forward-stepwise
# linear regression on the California Housing dataset. The target is
# MedHouseVal (median house value, $100k units). The available raw columns
# are exactly: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
# Latitude, Longitude.
#
# Propose exactly 10 engineered features that may carry predictive power.
# Rules:
#   * Use ONLY the raw columns listed above.
#   * Each feature must be a deterministic function of the raw columns.
#   * Briefly justify each suggestion (one sentence).
# Respond as JSON: a list of objects, each with keys "name", "formula",
# "rationale".
# ----------------------------------------------------------------------------
#
# --- RESPONSE (JSON) --------------------------------------------------------
# [
#   {"name": "rooms_per_person",  "formula": "AveRooms / AveOccup",
#    "rationale": "Living space per occupant — strong housing-quality signal."},
#   {"name": "bedrooms_per_room", "formula": "AveBedrms / AveRooms",
#    "rationale": "Bedroom share; lower values indicate larger living rooms / luxury."},
#   {"name": "log_medinc",        "formula": "log(MedInc)",
#    "rationale": "Income effects on price are typically concave."},
#   {"name": "medinc_sq",         "formula": "MedInc ** 2",
#    "rationale": "Captures convex high-income premium that linear MedInc misses."},
#   {"name": "log_population",    "formula": "log(Population)",
#    "rationale": "Population is right-skewed; log stabilizes the relationship."},
#   {"name": "pop_per_occup",     "formula": "Population / AveOccup",
#    "rationale": "Approximate household count in the block — neighborhood density."},
#   {"name": "dist_to_LA",        "formula": "sqrt((Latitude-34.05)^2 + (Longitude+118.24)^2)",
#    "rationale": "Proximity to Los Angeles strongly drives California prices."},
#   {"name": "dist_to_SF",        "formula": "sqrt((Latitude-37.77)^2 + (Longitude+122.42)^2)",
#    "rationale": "Proximity to San Francisco is a second major price gradient."},
#   {"name": "medinc_x_houseage", "formula": "MedInc * HouseAge",
#    "rationale": "Old homes in rich areas (historic neighborhoods) price differently."},
#   {"name": "lat_x_long",        "formula": "Latitude * Longitude",
#    "rationale": "Cheap nonlinear location term that flexibly captures geography."}
# ]
# ----------------------------------------------------------------------------

print()
print("=" * 72)
print("MODEL B — AI-augmented stepwise (8 raw + 10 LLM-suggested features)")
print("=" * 72)


def add_ai_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rooms_per_person"]  = out["AveRooms"] / out["AveOccup"]
    out["bedrooms_per_room"] = out["AveBedrms"] / out["AveRooms"]
    out["log_medinc"]        = np.log(out["MedInc"])
    out["medinc_sq"]         = out["MedInc"] ** 2
    out["log_population"]    = np.log(out["Population"])
    out["pop_per_occup"]     = out["Population"] / out["AveOccup"]
    out["dist_to_LA"]        = np.sqrt((out["Latitude"] - 34.05) ** 2 + (out["Longitude"] + 118.24) ** 2)
    out["dist_to_SF"]        = np.sqrt((out["Latitude"] - 37.77) ** 2 + (out["Longitude"] + 122.42) ** 2)
    out["medinc_x_houseage"] = out["MedInc"] * out["HouseAge"]
    out["lat_x_long"]        = out["Latitude"] * out["Longitude"]
    return out


X_train_B, X_test_B, dropped_B = drop_bad_columns(add_ai_features(X_train), add_ai_features(X_test))
if dropped_B:
    print(f"Dropped (NaN/Inf): {dropped_B}")
selected_B = forward_stepwise_aic(X_train_B, y_train)
rmse_B = fit_and_rmse(X_train_B, y_train, X_test_B, y_test, selected_B)
candidates_B = X_train_B.shape[1]
print(f"Candidate features: {candidates_B}")
print(f"Selected ({len(selected_B)}): {selected_B}")
print(f"Test RMSE: {rmse_B:.4f}")


# =============================================================================
# Model C — Kitchen-sink
# =============================================================================
print()
print("=" * 72)
print("MODEL C — Kitchen-sink stepwise (8 raw + 28 pairwise + 8 squares)")
print("=" * 72)


def add_kitchen_sink(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for a, b in itertools.combinations(RAW_COLS, 2):
        out[f"{a}_x_{b}"] = out[a] * out[b]
    for a in RAW_COLS:
        out[f"{a}_sq"] = out[a] ** 2
    return out


X_train_C, X_test_C, dropped_C = drop_bad_columns(add_kitchen_sink(X_train), add_kitchen_sink(X_test))
if dropped_C:
    print(f"Dropped (NaN/Inf): {dropped_C}")
selected_C = forward_stepwise_aic(X_train_C, y_train)
rmse_C = fit_and_rmse(X_train_C, y_train, X_test_C, y_test, selected_C)
candidates_C = X_train_C.shape[1]
print(f"Candidate features: {candidates_C}")
print(f"Selected ({len(selected_C)}): {selected_C}")
print(f"Test RMSE: {rmse_C:.4f}")


# =============================================================================
# Comparison + interpretation
# =============================================================================
print()
print("=" * 72)
print("COMPARISON")
print("=" * 72)
summary = pd.DataFrame(
    {
        "candidates_considered": [candidates_A, candidates_B, candidates_C],
        "features_selected":     [len(selected_A), len(selected_B), len(selected_C)],
        "test_RMSE":             [rmse_A, rmse_B, rmse_C],
    },
    index=["Model A (baseline)", "Model B (AI-augmented)", "Model C (kitchen-sink)"],
)
print(summary.round(4))

rmses = {"A": rmse_A, "B": rmse_B, "C": rmse_C}
winner = min(rmses, key=rmses.get)
b_vs_a = rmse_A - rmse_B
b_vs_c = rmse_B - rmse_C

print()
print("=" * 72)
print("INTERPRETATION")
print("=" * 72)
print(f"- Lowest test RMSE: Model {winner} ({rmses[winner]:.4f}).")
if b_vs_a > 0:
    print(f"- Model B beat Model A by {b_vs_a:.4f} RMSE "
          f"({100 * b_vs_a / rmse_A:.2f}% improvement).")
else:
    print(f"- Model B did NOT beat Model A (B was {-b_vs_a:.4f} worse).")
if abs(b_vs_c) < 0.01:
    print(f"- Model B essentially matched Model C using "
          f"{candidates_B}/{candidates_C} candidates.")
elif b_vs_c < 0:
    print(f"- Model B BEAT Model C by {-b_vs_c:.4f} RMSE despite using only "
          f"{candidates_B}/{candidates_C} candidates.")
else:
    print(f"- Model C beat Model B by {b_vs_c:.4f} RMSE.")
