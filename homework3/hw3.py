"""
DATA 419 Homework #3: What's in a name?

Predicts P(female = 1) for each first name in test.csv using logistic
regression on features engineered from the name string, and writes
submission.csv.

Usage:
    python hw3.py            # expects train.csv and test.csv in the current directory
    python hw3.py DATA_DIR   # or in DATA_DIR
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 419
VOWELS = set("aeiouy")

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")

print(f"train: {train.shape}, test: {test.shape}")
print(f"fraction female in train: {train['female'].mean():.3f}")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def clean(name):
    """Lowercase and keep only letters ("Mary-Kate" -> "marykate")."""
    return "".join(c for c in str(name).lower() if c.isalpha())


def make_features(names):
    """Turn a Series of raw names into a DataFrame of engineered features."""
    s = names.map(clean)
    feats = pd.DataFrame(index=names.index)

    # The whole name padded with ^ and $ so that character n-grams can tell
    # "starts with" / "ends with" apart from "contains" (e.g. "a$" = ends in a).
    feats["padded"] = "^" + s + "$"

    # Categorical: prefixes and suffixes of several lengths.
    for k in (1, 2, 3):
        feats[f"first{k}"] = s.str[:k]
        feats[f"last{k}"] = s.str[-k:]

    # Numeric.
    feats["length"] = s.str.len()
    feats["length_sq"] = feats["length"] ** 2  # lets the effect of length be non-monotone
    feats["num_vowels"] = s.map(lambda w: sum(c in VOWELS for c in w))
    feats["prop_vowels"] = feats["num_vowels"] / feats["length"].clip(lower=1)
    feats["ends_in_vowel"] = s.str[-1:].isin(list(VOWELS)).astype(int)
    feats["double_letter"] = s.map(
        lambda w: int(any(a == b for a, b in zip(w, w[1:])))
    )
    return feats


X_train = make_features(train["name"])
y_train = train["female"].to_numpy()
X_test = make_features(test["name"])

NUMERIC = ["length", "length_sq", "num_vowels", "prop_vowels", "ends_in_vowel", "double_letter"]
PREFIX = ["first1", "first2", "first3"]
SUFFIX = ["last1", "last2", "last3"]


def ngrams(lo, hi):
    """Binary indicators for every character n-gram (lo..hi) in the padded name."""
    return CountVectorizer(analyzer="char", ngram_range=(lo, hi), binary=True,
                           lowercase=False)


def onehot():
    return OneHotEncoder(handle_unknown="ignore")


def numeric():
    return StandardScaler()


def make_logreg(C=1.0, l1_ratio=0.0):
    """LogisticRegression that works on both older and newer scikit-learn."""
    major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
    if l1_ratio == 0.0:
        return LogisticRegression(C=C, max_iter=5000)
    if (major, minor) >= (1, 8):
        return LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=5000)
    return LogisticRegression(C=C, l1_ratio=l1_ratio, penalty="elasticnet",
                              solver="saga", max_iter=5000)


def model(transformers, C=1.0, l1_ratio=0.0):
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("logreg", make_logreg(C, l1_ratio)),
    ])


# ---------------------------------------------------------------------------
# Compare candidate feature sets with 5-fold cross-validation (log loss).
# Every encoder/scaler lives inside the Pipeline, so it is refit on each
# training fold and never sees the validation fold.
# ---------------------------------------------------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def cv_logloss(pipe):
    return -cross_val_score(pipe, X_train, y_train, cv=cv,
                            scoring="neg_log_loss").mean()


p = y_train.mean()
print("\nCross-validated log loss (lower is better)")
print(f"{'all 0.5':55s} {np.log(2):.4f}")
print(f"{'constant = training fraction female':55s} "
      f"{-(p * np.log(p) + (1 - p) * np.log(1 - p)):.4f}")

candidates = {
    "last letter only":
        [("suf", onehot(), ["last1"])],
    "last 1-2 letters":
        [("suf", onehot(), ["last1", "last2"])],
    "last 1-3 letters":
        [("suf", onehot(), SUFFIX)],
    "last 1-3 + first 1-3":
        [("suf", onehot(), SUFFIX), ("pre", onehot(), PREFIX)],
    "last 1-3 + first 1-3 + numeric":
        [("suf", onehot(), SUFFIX), ("pre", onehot(), PREFIX),
         ("num", numeric(), NUMERIC)],
    "char 1-3 grams (padded)":
        [("ng", ngrams(1, 3), "padded")],
    "char 1-4 grams (padded)":
        [("ng", ngrams(1, 4), "padded")],
    "char 1-5 grams (padded)":
        [("ng", ngrams(1, 5), "padded")],
    "char 1-4 grams + suffixes + numeric":
        [("ng", ngrams(1, 4), "padded"), ("suf", onehot(), SUFFIX),
         ("num", numeric(), NUMERIC)],
    "char 1-5 grams + suffixes + numeric":
        [("ng", ngrams(1, 5), "padded"), ("suf", onehot(), SUFFIX),
         ("num", numeric(), NUMERIC)],
}

results = {}
for label, transformers in candidates.items():
    results[label] = cv_logloss(model(transformers))
    print(f"{label:55s} {results[label]:.4f}")

best_label = min(results, key=results.get)
print(f"\nBest feature set at default C=1: {best_label}")


# ---------------------------------------------------------------------------
# Tune regularization (C and L1/L2 mix) for the best feature set.
# ---------------------------------------------------------------------------

grid = GridSearchCV(
    model(candidates[best_label]),
    param_grid={"logreg__C": [0.03, 0.1, 0.3, 1, 3, 10]},
    scoring="neg_log_loss", cv=cv, n_jobs=-1,
)
grid.fit(X_train, y_train)
best_C = grid.best_params_["logreg__C"]
print("\nRidge (L2) regularization:")
for C, score in zip(grid.cv_results_["param_logreg__C"],
                    grid.cv_results_["mean_test_score"]):
    print(f"  C = {C:<6} log loss = {-score:.4f}")

best_l1, best_score = 0.0, -grid.best_score_
print("\nElastic net at the best C:")
for l1 in (0.5, 1.0):
    score = cv_logloss(model(candidates[best_label], C=best_C, l1_ratio=l1))
    print(f"  l1_ratio = {l1:<4} log loss = {score:.4f}")
    if score < best_score:
        best_l1, best_score = l1, score

print(f"\nFinal choice: {best_label}, C = {best_C}, l1_ratio = {best_l1} "
      f"(CV log loss {best_score:.4f})")


# ---------------------------------------------------------------------------
# Fit the final model on all of train.csv and inspect it.
# ---------------------------------------------------------------------------

final = model(candidates[best_label], C=best_C, l1_ratio=best_l1)
final.fit(X_train, y_train)
assert list(final.classes_) == [0, 1]

try:
    names = final.named_steps["features"].get_feature_names_out()
    coefs = pd.Series(final.named_steps["logreg"].coef_[0], index=names)
    coefs = coefs.sort_values()
    print("\nMost 'male' features:\n", coefs.head(15).to_string())
    print("\nMost 'female' features:\n", coefs.tail(15).to_string())
except Exception as e:  # feature names are a nicety, not essential
    print(f"(could not list feature names: {e})")


# ---------------------------------------------------------------------------
# Predict and write submission.csv
# ---------------------------------------------------------------------------

p_female = final.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"id": test["id"], "female": p_female})
submission.to_csv("submission.csv", index=False)

assert submission.shape == (len(test), 2)
assert submission["id"].is_unique
assert submission["female"].between(0, 1).all()
print("\n", submission.head())
print(submission.shape)
print(submission["female"].describe())
print("\nWrote submission.csv")
