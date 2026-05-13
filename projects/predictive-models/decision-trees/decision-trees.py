"""
Decision Trees — ID3 (depth 2) + sklearn comparison
====================================================
Charlie McChesney  ·  ECON 475 Section 2

A from-scratch implementation of entropy, information gain, and the ID3
algorithm at depth 2, alongside a comparison against sklearn's
DecisionTreeClassifier at varying depths.

Run:
    pip install -r requirements.txt
    python decision-trees.py
"""

import os
from math import log2

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for the CSV alongside the script first (the layout inside the .zip
# download), then fall back to a data/ subfolder (the layout used on the live
# site so the in-browser demo can fetch it via data/...csv).
_CSV = "section2-dataset.csv"
DATA_PATH = next(
    p for p in (os.path.join(SCRIPT_DIR, _CSV), os.path.join(SCRIPT_DIR, "data", _CSV))
    if os.path.exists(p)
)


# =============================================================================
# A. Entropy, Gini, & Information Gain
# =============================================================================

def entropy(t):
    """Shannon entropy of a pandas Series."""
    counts = t.value_counts()
    total = len(t)
    ent = 0.0
    for count in counts:
        p = count / total
        if p > 0:
            ent -= p * log2(p)
    return ent


def gini(t):
    """Gini impurity of a pandas Series."""
    counts = t.value_counts()
    total = len(t)
    gi = 0.0
    for count in counts:
        p = count / total
        gi += p ** 2
    return 1 - gi


def rem(d, t):
    """Remaining entropy after splitting target t on descriptive feature d."""
    total = len(t)
    rem_val = 0.0
    for val in d.unique():
        subset = t[d == val]
        weight = len(subset) / total
        rem_val += weight * entropy(subset)
    return rem_val


def maxIG(df, t):
    """Return the column name with the highest information gain w.r.t. t."""
    base_entropy = entropy(t)
    best_feature, best_ig = None, -1
    for col in df.columns:
        ig = base_entropy - rem(df[col], t)
        if ig > best_ig:
            best_ig, best_feature = ig, col
    return best_feature


# =============================================================================
# B. ID3 Algorithm (depth 2) & Prediction
# =============================================================================

def dTree(df, t):
    """Build a depth-2 ID3 decision tree as a nested dict."""
    root_feature = maxIG(df, t)
    tree = {root_feature: {}}

    for root_val in df[root_feature].unique():
        mask = df[root_feature] == root_val
        sub_df = df[mask].drop(columns=[root_feature])
        sub_t = t[mask]

        if sub_t.nunique() == 1 or sub_df.empty:
            tree[root_feature][root_val] = sub_t.mode()[0]
        else:
            second_feature = maxIG(sub_df, sub_t)
            tree[root_feature][root_val] = {second_feature: {}}
            for second_val in sub_df[second_feature].unique():
                mask2 = sub_df[second_feature] == second_val
                sub_t2 = sub_t[mask2]
                if len(sub_t2) == 0:
                    tree[root_feature][root_val][second_feature][second_val] = t.mode()[0]
                else:
                    tree[root_feature][root_val][second_feature][second_val] = sub_t2.mode()[0]

    return tree


def predict(tree, obsDF):
    """Predict the target value for a single observation using the tree."""
    root_feature = list(tree.keys())[0]
    root_val = obsDF[root_feature].values[0]

    subtree = tree[root_feature].get(root_val)
    if not isinstance(subtree, dict):
        return subtree

    second_feature = list(subtree.keys())[0]
    second_val = obsDF[second_feature].values[0]
    prediction = subtree[second_feature].get(second_val)

    if prediction is None or isinstance(prediction, dict):
        leaves = [v for v in subtree[second_feature].values() if not isinstance(v, dict)]
        return leaves[0] if leaves else None
    return prediction


# =============================================================================
# C. Custom dTree accuracy (train/test split)
# =============================================================================

def dTAccuracy(df, t, random_state=2):
    ds_train, ds_test, t_train, t_test = train_test_split(
        df, t, test_size=0.3, random_state=random_state
    )
    tree = dTree(ds_train.reset_index(drop=True), t_train.reset_index(drop=True))

    train_correct = sum(
        predict(tree, ds_train.iloc[[i]].reset_index(drop=True)) == t_train.iloc[i]
        for i in range(len(ds_train))
    )
    test_correct = sum(
        predict(tree, ds_test.iloc[[i]].reset_index(drop=True)) == t_test.iloc[i]
        for i in range(len(ds_test))
    )
    return [train_correct / len(ds_train), test_correct / len(ds_test)]


# =============================================================================
# D. sklearn DecisionTreeClassifier comparison
# =============================================================================

def dTAccuracy2(df, t, max_depth=2, criterion="entropy", random_state=1):
    ds_train, ds_test, t_train, t_test = train_test_split(
        df, t, test_size=0.3, random_state=random_state
    )
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)
    clf.fit(ds_train, t_train)
    return [
        accuracy_score(t_train, clf.predict(ds_train)),
        accuracy_score(t_test, clf.predict(ds_test)),
    ]


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    target_col = "Committed Crime"
    t = data[target_col]
    df = data.drop(columns=[target_col])

    print("=" * 60)
    print("A. Entropy & Information Gain")
    print("=" * 60)
    print(f"Entropy of target: {entropy(t):.4f}")
    print(f"Gini    of target: {gini(t):.4f}")
    print(f"Feature with highest IG: {maxIG(df, t)}")

    print("\n" + "=" * 60)
    print("B. Custom ID3 Tree (depth 2)")
    print("=" * 60)
    tree = dTree(df, t)
    print(f"Tree: {tree}")
    obs = df.iloc[[0]]
    print(f"Prediction for row 0: {predict(tree, obs)} (actual: {t.iloc[0]})")

    print("\n" + "=" * 60)
    print("C. Custom ID3 accuracy (depth 2, 70/30 split)")
    print("=" * 60)
    train_acc, test_acc = dTAccuracy(df, t)
    print(f"Train: {train_acc:.4f}   Test: {test_acc:.4f}")

    print("\n" + "=" * 60)
    print("D. sklearn DecisionTreeClassifier (entropy, varying depth)")
    print("=" * 60)
    for depth in [2, 3, 4, 5]:
        train_acc, test_acc = dTAccuracy2(df, t, max_depth=depth)
        print(f"depth={depth}   Train: {train_acc:.4f}   Test: {test_acc:.4f}")
