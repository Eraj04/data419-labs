"""DATA 419 Homework 1: linear regression of World Cup match attendance.

Save in data419-labs/lab1 and activate your class virtual environment.
From data419-labs, install the corrected Homework 0 package and dependencies:
    python -m pip install -e ./world_cup
    python -m pip install numpy pandas pyarrow scikit-learn matplotlib scipy
    python -m world_cup.pull_all
Run the analysis:
    python lab1/world_cup_homework1.py

This script loads the Homework 0 package, fits the four required models, and
saves two plots and hw1_results.json next to this file in a hw1_results folder.
Variables, transformations, seed, and model definitions were chosen before
viewing test results. No model or predictor is tuned using test performance.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import world_cup

SEED = 12345
np.random.seed(SEED)
OUTPUT_DIR = Path(__file__).resolve().parent / "hw1_results"
NUMERIC = ["year", "participating_teams"]
PREDICTORS = NUMERIC + ["round"]


def prepare_data():
    """Create one row per match and join tournament-level predictors."""
    tables = world_cup.load()    
    matches = tables["matches"].copy()

    if "match_id" not in matches.columns:
        keys = ["date", "home_team", "away_team"]
        if matches[keys].isna().any().any() or matches.duplicated(keys).any():
            raise ValueError("Check for missing or duplicate match identifiers.")
        matches.insert(0, "match_id", range(len(matches)))

    tournaments = tables["tournaments"][["year", "teams"]].copy()
    # The participant list is tournament metadata. Attendance never enters it.
    participants = pd.concat([
        matches[["year", "home_team"]].rename(columns={"home_team": "team"}),
        matches[["year", "away_team"]].rename(columns={"away_team": "team"}),
    ]).drop_duplicates()
    counts = participants.groupby("year")["team"].nunique().rename("participating_teams")
    info = tournaments.merge(counts, on="year", validate="one_to_one")
    corrections = info.loc[info["teams"] != info["participating_teams"]].to_dict("records")
    data = matches.merge(info[["year", "participating_teams"]], on="year",
                         how="left", validate="many_to_one")
    selected = ["attendance", *PREDICTORS]
    for col in ["attendance", *NUMERIC]:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)
        data.loc[~np.isfinite(data[col]), col] = np.nan
    data["round"] = data["round"].astype("string").str.strip().replace("", pd.NA)
    invalid_attendance = int((data["attendance"] <= 0).sum())
    invalid_teams = int((data["participating_teams"] < 2).sum())
    data.loc[data["attendance"] <= 0, "attendance"] = np.nan
    data.loc[data["participating_teams"] < 2, "participating_teams"] = np.nan
    missing = data[selected].isna().sum().astype(int).to_dict()
    before = len(data)
    data = data.dropna(subset=selected).reset_index(drop=True)
    if data["match_id"].duplicated().any():
        raise ValueError("The modeling table must contain one row per match.")
    return data, {"source_rows": before, "modeling_rows": len(data),
                  "removed_rows": before-len(data), "missing_by_variable": missing,
                  "nonpositive_attendance_rows": invalid_attendance,
                  "invalid_team_count_rows": invalid_teams,
                  "team_count_corrections": corrections,
                  "round_counts": {str(k): int(v) for k, v in data["round"].value_counts().items()}}


def design_matrices(scaled, categorical):
    """Every design has an explicit intercept; only Model 1Q is quadratic."""
    ones = np.ones(len(scaled))
    x1, x2 = scaled[:, 0], scaled[:, 1]
    return {"Model 1": np.column_stack([ones, x1]),
            "Model 1Q": np.column_stack([ones, x1, x1**2]),
            "Model 2": np.column_stack([ones, x1, x2]),
            "Model 3": np.column_stack([ones, x1, x2, categorical])}


def scores(actual, predicted):
    return {"r2": float(r2_score(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted)))}


def make_plots(X_train, y_train, scaled_train, models, designs):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    comma = FuncFormatter(lambda value, _: f"{value:,.0f}")
    grid = np.linspace(scaled_train[:, 0].min(), scaled_train[:, 0].max(), 400)
    fig, ax = plt.subplots(figsize=(8.2, 3.9), layout="constrained")
    ax.scatter(scaled_train[:, 0], y_train, s=16, color="#3D6480", alpha=.42,
               linewidths=0, label=f"Training matches (n={len(y_train)})")
    ax.plot(grid, models["Model 1"].predict(np.column_stack([np.ones(len(grid)), grid])),
            color="green", lw=2.2, label="Model 1: linear")
    ax.plot(grid, models["Model 1Q"].predict(np.column_stack([np.ones(len(grid)), grid, grid**2])),
            color="red", lw=2.2, label="Model 1Q: quadratic")
    ax.set(xlabel="Tournament year, standardized using training data (z-score)",
           ylabel="Attendance (spectators)", title="World Cup attendance and tournament year")
    ax.yaxis.set_major_formatter(comma)
    ax.grid(alpha=.15)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.savefig(OUTPUT_DIR / "linear_quadratic.png", dpi=200)
    plt.close(fig)

    fitted = models["Model 3"].predict(designs["Model 3"])
    residuals = y_train - fitted
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), layout="constrained")
    axes[0].scatter(fitted, residuals, color="#3D6480", alpha=.45, s=15, linewidths=0)
    axes[0].axhline(0, color="#B33A3A", lw=1.2)
    axes[0].set(xlabel="Fitted attendance", ylabel="Actual - fitted attendance",
                title="Model 3: residuals vs. fitted")
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0].xaxis.set_major_formatter(comma)
    axes[0].yaxis.set_major_formatter(comma)
    (theory, ordered), (slope, intercept, qq_r) = stats.probplot(residuals, dist="norm")
    axes[1].scatter(theory, ordered, color="#3D6480", alpha=.6, s=13, linewidths=0)
    axes[1].plot(theory, intercept+slope*theory, color="#B33A3A", lw=1.3)
    axes[1].set(xlabel="Theoretical normal quantile", ylabel="Ordered residual (spectators)",
                title="Model 3: normal Q-Q plot")
    axes[1].yaxis.set_major_formatter(comma)
    for ax in axes:
        ax.grid(alpha=.15)
        ax.tick_params(labelsize=8)
    fig.savefig(OUTPUT_DIR / "model3_diagnostics.png", dpi=200)
    plt.close(fig)
    r = float(X_train[NUMERIC].corr().iloc[0, 1])
    return {"numeric_predictor_correlation_train": r, "pairwise_vif": float(1/(1-r*r)),
            "residual_skew_train": float(stats.skew(residuals, bias=False)),
            "qq_correlation_train": float(qq_r),
            "largest_absolute_training_residual": float(np.abs(residuals).max())}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, preparation = prepare_data()
    X = data[PREDICTORS].copy()
    y = data["attendance"].to_numpy(dtype=float)  # Do not scale the response.
    # Exactly one split, before fitting any preprocessing or regression.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=SEED)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train[NUMERIC])
    test_scaled = scaler.transform(X_test[NUMERIC])
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    train_categories = encoder.fit_transform(X_train[["round"]])
    test_categories = encoder.transform(X_test[["round"]])
    train_design = design_matrices(train_scaled, train_categories)
    test_design = design_matrices(test_scaled, test_categories)
    names = {"Model 1": ["intercept", "z_year"],
             "Model 1Q": ["intercept", "z_year", "z_year_squared"],
             "Model 2": ["intercept", "z_year", "z_participating_teams"],
             "Model 3": ["intercept", "z_year", "z_participating_teams",
                         *encoder.get_feature_names_out(["round"]).tolist()]}
    models, results = {}, {}
    for name, matrix in train_design.items():
        if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
            raise ValueError(f"{name}: redundant design columns.")
        model = LinearRegression(fit_intercept=False).fit(matrix, y_train)
        models[name] = model
        results[name] = {"train": scores(y_train, model.predict(matrix)),
                         "test": scores(y_test, model.predict(test_design[name])),
                         "coefficients": dict(zip(names[name], model.coef_.tolist())),
                         "design_columns": matrix.shape[1]}

    # Verify the assignment's split, scaling, and explicit-intercept requirements.
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train)+len(X_test) == len(data)
    assert np.allclose(train_scaled.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(train_scaled.std(axis=0), 1)
    assert np.allclose(scaler.mean_, X_train[NUMERIC].mean().to_numpy())
    for name in models:
        assert np.all(train_design[name][:, 0] == 1)
        assert np.all(test_design[name][:, 0] == 1)
        assert len(test_design[name]) == len(y_test)
    diagnostics = make_plots(X_train, y_train, train_scaled, models, train_design)
    metadata = {"seed": SEED, "preparation": preparation,
                "train_rows": len(X_train), "test_rows": len(X_test),
                "scaler_mean": dict(zip(NUMERIC, scaler.mean_.tolist())),
                "scaler_sd": dict(zip(NUMERIC, scaler.scale_.tolist())),
                "training_round_categories": encoder.categories_[0].tolist(),
                "reference_round": str(encoder.categories_[0][0]),
                "unseen_test_rounds": sorted(set(X_test["round"])-set(X_train["round"])),
                "models": results, "diagnostics": diagnostics,
                "train_mean_baseline_test": scores(y_test, np.full(len(y_test), y_train.mean())),
                "training_match_ids": data.loc[X_train.index, "match_id"].tolist(),
                "test_match_ids": data.loc[X_test.index, "match_id"].tolist(),
                "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                             "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                             "scipy": scipy.__version__}}
    (OUTPUT_DIR / "hw1_results.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Rows: {len(data)}; training: {len(X_train)}; test: {len(X_test)}")
    print(f"Dropped rows: {preparation['removed_rows']}")
    print(f"Team-count corrections: {preparation['team_count_corrections']}")
    print(f"One-hot reference round: {metadata['reference_round']}")
    print(f"Training means: {metadata['scaler_mean']}")
    print(f"Training standard deviations: {metadata['scaler_sd']}")
    print(f"{'Model':<12} {'Train R2':>10} {'Test R2':>10} {'Train RMSE':>14} {'Test RMSE':>14}")
    for name, result in results.items():
        a, b = result["train"], result["test"]
        print(f"{name:<12} {a['r2']:>10.4f} {b['r2']:>10.4f} {a['rmse']:>14,.2f} {b['rmse']:>14,.2f}")
    b0, b1 = models["Model 1"].coef_
    print(f"Model 1: predicted attendance = {b0:.4f} + ({b1:.4f}) * z_year")
    for new, old in [("Model 1Q", "Model 1"), ("Model 2", "Model 1"), ("Model 3", "Model 2")]:
        diff = results[new]["test"]["r2"]-results[old]["test"]["r2"]
        drop = results[old]["test"]["rmse"]-results[new]["test"]["rmse"]
        print(f"{new} vs {old}: test R2 change {100*diff:+.2f} percentage points; "
              f"RMSE reduction {drop:+,.2f} spectators")
    print(f"Training diagnostics: {diagnostics}")
    print(f"Results saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
