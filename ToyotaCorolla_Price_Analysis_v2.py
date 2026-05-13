"""
================================================================================
ToyotaCorolla_Price_Analysis_v2.py
================================================================================
Toyota Corolla Resale Price Prediction
Machine Learning Analysis for Used-Car Dealership Pricing

Purpose
-------
Estimate the resale price of refurbished Toyota Corollas using data science
techniques, helping dealerships assess potential profit from used-car inventory.

Pipeline
--------
1. Data Loading & Cleaning       – remove non-numeric columns, handle NaN
2. Exploratory Data Analysis     – descriptive statistics, correlation study
3. Normalization                 – MinMaxScaler → normalized DataFrame
4. Visualization                 – scatter plots & box plots
5. Decision Tree Regression      – cross-validated price prediction model

Dataset
-------
ToyotaCorolla.csv  (place in the same directory as this file)
Source : https://www.kaggle.com/datasets/klkwak/toyotacorollacsv
Course : https://open.hpi.de/courses/datascience2023/overview

Usage
-----
    # Run the full analysis (generates plots inline / saves to /outputs)
    python ToyotaCorolla_Price_Analysis_v2.py

    # Import individual components in another script or notebook
    from ToyotaCorolla_Price_Analysis_v2 import load_and_clean, train_model

Requirements
------------
    pip install pandas numpy matplotlib seaborn scikit-learn

================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Plot defaults ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 5)})

# ── Output directory (created automatically) ─────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CSV_PATH         = "ToyotaCorolla.csv"
TARGET           = "Price"
NON_NUMERIC_COLS = ["Id", "Model", "Fuel_Type"]
RANDOM_STATE     = 42
TEST_SIZE        = 0.20
CV_FOLDS         = 5
MAX_DEPTH_RANGE  = range(1, 16)


# ==============================================================================
# 1. DATA LOADING & CLEANING
# ==============================================================================

def load_and_clean(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load the Toyota Corolla CSV and return a clean, fully numeric DataFrame.

    Steps
    -----
    - Drop identifier and free-text columns (Id, Model, Fuel_Type)
    - Keep only numeric (integer / float) columns
    - Coerce whole-number floats to int dtype
    - Drop rows containing NaN values

    Parameters
    ----------
    csv_path : str
        Path to ToyotaCorolla.csv

    Returns
    -------
    pd.DataFrame
        Clean DataFrame ready for analysis
    """
    df_raw = pd.read_csv(csv_path)
    print(f"[load]  Raw shape     : {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    # Drop non-numeric columns
    drop_cols = [c for c in NON_NUMERIC_COLS if c in df_raw.columns]
    df = df_raw.drop(columns=drop_cols)

    # Keep only numeric dtypes
    df = df.select_dtypes(include=[np.number])

    # Promote whole-number floats → int
    for col in df.columns:
        if df[col].dtype == float:
            if df[col].dropna().apply(lambda x: x == int(x)).all():
                df[col] = df[col].astype("Int64").astype(int)

    # Remove NaN rows
    rows_before = len(df)
    df.dropna(inplace=True)
    print(f"[clean] Dropped cols  : {drop_cols}")
    print(f"[clean] Dropped rows  : {rows_before - len(df)} (NaN)")
    print(f"[clean] Final shape   : {df.shape[0]} rows × {df.shape[1]} cols\n")

    return df.reset_index(drop=True)


# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ==============================================================================

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and print descriptive statistics for every column.

    Includes count, mean, std, min/max, quartiles, range, and the
    coefficient of variation (CV = std / mean) as a measure of relative spread.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Transposed stats table with an extra 'range' and 'cv' column
    """
    stats = df.describe().T
    stats["range"] = stats["max"] - stats["min"]
    stats["cv"]    = (stats["std"] / stats["mean"]).round(3)

    pd.set_option("display.float_format", "{:,.2f}".format)
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(stats[["count", "mean", "std", "min", "25%", "50%",
                  "75%", "max", "range", "cv"]].to_string())
    print()
    return stats


def correlation_analysis(df: pd.DataFrame) -> pd.Series:
    """
    Compute the full correlation matrix and print correlations with Price,
    sorted by absolute magnitude.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series
        Pearson correlations of all features with Price (sorted descending)
    """
    corr_matrix = df.corr()
    price_corr = (
        corr_matrix[TARGET]
        .drop(TARGET)
        .reindex(corr_matrix[TARGET].drop(TARGET).abs().sort_values(ascending=False).index)
    )

    print("=" * 60)
    print("CORRELATION WITH PRICE (sorted by |r|)")
    print("=" * 60)
    print(price_corr.to_string())
    print()
    return price_corr


# ==============================================================================
# 3. NORMALIZATION
# ==============================================================================

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Min-Max normalization to all columns, scaling values to [0, 1].

    The result is stored in a variable named `normalized` to match the
    convention pd.DataFrame.normalized used in the notebook version.

    Note: Decision Trees are scale-invariant; normalization here is for
    visual EDA comparability and future model compatibility.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame (all values in [0, 1])
    """
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df)
    normalized = pd.DataFrame(arr, columns=df.columns)

    print("=" * 60)
    print("NORMALIZATION  (MinMaxScaler → [0, 1])")
    print("=" * 60)
    print(normalized.describe().loc[["min", "max"]].T.to_string())
    print()
    return normalized


# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================

def _save(fig: plt.Figure, filename: str) -> None:
    """Save a figure to the outputs directory and close it."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot]  Saved → {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Render a lower-triangle heatmap of the full correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
    """
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.4,
        ax=ax, annot_kws={"size": 7},
    )
    ax.set_title("Full Correlation Heatmap — Toyota Corolla Dataset",
                 fontsize=14, pad=14)
    plt.tight_layout()
    _save(fig, "01_correlation_heatmap.png")


def plot_top_correlations(price_corr: pd.Series) -> None:
    """
    Horizontal bar chart of the top-12 features correlated with Price.

    Blue bars = positive correlation (price rises with feature).
    Red  bars = negative correlation (price falls as feature rises).

    Parameters
    ----------
    price_corr : pd.Series
        Output of correlation_analysis()
    """
    top = price_corr.abs().sort_values(ascending=False).head(12)
    colors = ["#d7191c" if price_corr[f] < 0 else "#2166ac" for f in top.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Absolute Pearson Correlation with Price")
    ax.set_title("Top 12 Features Correlated with Sale Price", fontsize=13)
    ax.axvline(0.3, color="grey", linestyle="--", linewidth=0.8,
               label="r = 0.3 threshold")
    ax.legend(fontsize=9)

    for bar, feat in zip(bars[::-1], top.index[::-1]):
        direction = "▲" if price_corr[feat] > 0 else "▼"
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                direction, va="center", fontsize=9,
                color="#2166ac" if price_corr[feat] > 0 else "#d7191c")

    plt.tight_layout()
    _save(fig, "02_top_correlations.png")


def plot_scatter_raw(df: pd.DataFrame, price_corr: pd.Series) -> None:
    """
    Scatter plots (raw values) for the 6 most correlated continuous features
    vs. Price, with a linear trend line and Pearson r annotation.

    Parameters
    ----------
    df         : pd.DataFrame  — clean dataset
    price_corr : pd.Series     — correlations with Price
    """
    continuous = [
        f for f in price_corr.abs().sort_values(ascending=False).index
        if df[f].nunique() > 10
    ][:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feat in zip(axes.flatten(), continuous):
        r = price_corr[feat]
        ax.scatter(df[feat], df[TARGET], alpha=0.3, s=15,
                   color="steelblue", edgecolors="none")
        z = np.polyfit(df[feat], df[TARGET], 1)
        xs = np.linspace(df[feat].min(), df[feat].max(), 200)
        ax.plot(xs, np.poly1d(z)(xs), "r--", linewidth=1.5, label=f"r = {r:.2f}")
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("Price (€)", fontsize=10)
        ax.set_title(f"{feat} vs Price", fontsize=11)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Scatter Plots: Key Features vs. Sale Price",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "03_scatter_raw.png")


def plot_scatter_normalized(normalized: pd.DataFrame,
                             price_corr: pd.Series,
                             df: pd.DataFrame) -> None:
    """
    Same scatter plots as plot_scatter_raw but on the normalized [0, 1] scale,
    allowing direct slope comparison across features.

    Parameters
    ----------
    normalized : pd.DataFrame  — output of normalize()
    price_corr : pd.Series     — correlations with Price (unchanged by scaling)
    df         : pd.DataFrame  — clean dataset (used to identify continuous cols)
    """
    continuous = [
        f for f in price_corr.abs().sort_values(ascending=False).index
        if df[f].nunique() > 10
    ][:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feat in zip(axes.flatten(), continuous):
        r = price_corr[feat]
        ax.scatter(normalized[feat], normalized[TARGET], alpha=0.3, s=15,
                   color="darkorange", edgecolors="none")
        z = np.polyfit(normalized[feat], normalized[TARGET], 1)
        xs = np.linspace(0, 1, 200)
        ax.plot(xs, np.poly1d(z)(xs), "navy", linestyle="--",
                linewidth=1.5, label=f"r = {r:.2f}")
        ax.set_xlabel(f"{feat} (normalized)", fontsize=9)
        ax.set_ylabel("Price (normalized)", fontsize=9)
        ax.set_title(f"{feat} vs Price (norm.)", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Scatter Plots on Normalized Data", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "04_scatter_normalized.png")


def plot_boxplots_binary(df: pd.DataFrame, price_corr: pd.Series) -> None:
    """
    Box plots of Price split by the 12 most correlated binary (0/1) features.

    Each subplot shows two boxes: feature absent (0) vs. feature present (1),
    revealing the price premium associated with each optional feature.

    Parameters
    ----------
    df         : pd.DataFrame
    price_corr : pd.Series
    """
    binary_cols = [
        c for c in df.columns
        if c != TARGET
        and df[c].nunique() == 2
        and set(df[c].unique()).issubset({0, 1})
    ]
    top_binary = [
        f for f in price_corr.abs().sort_values(ascending=False).index
        if f in binary_cols
    ][:12]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, feat in zip(axes.flatten(), top_binary):
        data = [df.loc[df[feat] == 0, TARGET], df.loc[df[feat] == 1, TARGET]]
        ax.boxplot(
            data,
            labels=[f"{feat}=0", f"{feat}=1"],
            patch_artist=True,
            boxprops=dict(facecolor="#AED6F1"),
            medianprops=dict(color="#E74C3C", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )
        ax.set_title(f"{feat}  (r={price_corr[feat]:.2f})", fontsize=9)
        ax.set_ylabel("Price (€)", fontsize=8)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.tick_params(labelsize=8)

    fig.suptitle("Box Plots: Price Distribution by Binary Feature Presence",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, "05_boxplots_binary.png")


def plot_boxplots_age_groups(df: pd.DataFrame) -> None:
    """
    Box plots of Price grouped by car age (binned into 6-month intervals).

    Illustrates the depreciation pattern across the vehicle's lifespan.

    Parameters
    ----------
    df : pd.DataFrame
    """
    df_plot = df.copy()
    df_plot["Age_Group"] = pd.cut(
        df_plot["Age_08_04"],
        bins=[0, 12, 24, 36, 48, 60, 120],
        labels=["0–12m", "13–24m", "25–36m", "37–48m", "49–60m", "60m+"],
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    df_plot.boxplot(
        column=TARGET, by="Age_Group", ax=ax,
        patch_artist=True,
        boxprops=dict(facecolor="#A9DFBF"),
        medianprops=dict(color="#C0392B", linewidth=2),
    )
    ax.set_title("Price Distribution by Car Age Group", fontsize=13)
    ax.set_xlabel("Age Group (months since manufacture)")
    ax.set_ylabel("Price (€)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.suptitle("")
    plt.tight_layout()
    _save(fig, "06_boxplots_age_groups.png")


# ==============================================================================
# 5. DECISION TREE REGRESSION
# ==============================================================================

def find_best_depth(X_train: pd.DataFrame,
                    y_train: pd.Series) -> int:
    """
    Select the optimal tree depth via 5-fold cross-validated MAE.

    Tries every integer depth from 1 to 15 and returns the depth that
    minimises average cross-validated Mean Absolute Error on the training set.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    int
        Best max_depth value
    """
    depths     = list(MAX_DEPTH_RANGE)
    cv_scores  = []

    for d in depths:
        dt     = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE)
        scores = cross_val_score(dt, X_train, y_train,
                                 cv=CV_FOLDS, scoring="neg_mean_absolute_error")
        cv_scores.append(-scores.mean())

    best_depth = depths[int(np.argmin(cv_scores))]

    # ── plot depth vs CV-MAE ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(depths, cv_scores, marker="o", color="steelblue", linewidth=2)
    ax.axvline(best_depth, color="red", linestyle="--",
               label=f"Best depth = {best_depth}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel(f"CV MAE (€)  [{CV_FOLDS}-fold]")
    ax.set_title("Decision Tree — Choosing Optimal Tree Depth")
    ax.legend()
    plt.tight_layout()
    _save(fig, "07_cv_depth_selection.png")

    print(f"[model] Best max_depth : {best_depth}  "
          f"(CV MAE = €{cv_scores[best_depth - 1]:,.0f})\n")
    return best_depth


def train_model(df: pd.DataFrame) -> tuple:
    """
    Train a Decision Tree Regressor and return the fitted model alongside
    the train/test splits and performance metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Clean dataset (output of load_and_clean)

    Returns
    -------
    tuple : (model, X_train, X_test, y_train, y_test, features)
        - model    : fitted DecisionTreeRegressor
        - X_train  : training feature matrix
        - X_test   : test feature matrix
        - y_train  : training target vector
        - y_test   : test target vector
        - features : list of feature names used
    """
    features = [c for c in df.columns if c != TARGET]
    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[split] Train : {len(X_train)} samples  |  "
          f"Test : {len(X_test)} samples  |  "
          f"Features : {len(features)}\n")

    best_depth = find_best_depth(X_train, y_train)

    model = DecisionTreeRegressor(max_depth=best_depth, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, features


def evaluate_model(model: DecisionTreeRegressor,
                   X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Compute and print MAE, RMSE, and R² on both train and test splits.

    Parameters
    ----------
    model   : fitted DecisionTreeRegressor
    X_train, X_test : feature matrices
    y_train, y_test : target vectors

    Returns
    -------
    dict
        {"train_mae", "test_mae", "train_rmse", "test_rmse",
         "train_r2", "test_r2", "y_pred_test"}
    """
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    results = {
        "train_mae"   : mean_absolute_error(y_train, y_pred_train),
        "test_mae"    : mean_absolute_error(y_test,  y_pred_test),
        "train_rmse"  : np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "test_rmse"   : np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        "train_r2"    : r2_score(y_train, y_pred_train),
        "test_r2"     : r2_score(y_test,  y_pred_test),
        "y_pred_test" : y_pred_test,
    }

    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    for split in ("train", "test"):
        print(f"  {split.capitalize():<6}  "
              f"MAE = €{results[f'{split}_mae']:,.0f}  |  "
              f"RMSE = €{results[f'{split}_rmse']:,.0f}  |  "
              f"R² = {results[f'{split}_r2']:.4f}")
    print()
    return results


def plot_predicted_vs_actual(y_test: pd.Series,
                              y_pred_test: np.ndarray) -> None:
    """
    Scatter plot of predicted vs. actual prices on the test set.

    Points lying on the red dashed diagonal indicate perfect predictions.
    Systematic deviations reveal bias; vertical spread reveals variance.

    Parameters
    ----------
    y_test      : pd.Series    — actual prices
    y_pred_test : np.ndarray   — model predictions
    """
    lims = [
        min(y_test.min(), y_pred_test.min()) - 500,
        max(y_test.max(), y_pred_test.max()) + 500,
    ]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred_test, alpha=0.45, s=25,
               color="steelblue", label="Test predictions")
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Price (€)")
    ax.set_ylabel("Predicted Price (€)")
    ax.set_title("Decision Tree: Predicted vs Actual Price (Test Set)",
                 fontsize=12)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    plt.tight_layout()
    _save(fig, "08_predicted_vs_actual.png")


def plot_residuals(y_test: pd.Series, y_pred_test: np.ndarray) -> None:
    """
    Two-panel residuals diagnostic:
      Left  — residuals vs. predicted (checks for heteroscedasticity)
      Right — histogram of residuals (checks for symmetry / bias)

    Ideal behaviour: points scatter randomly around zero; histogram is
    roughly bell-shaped and centred at zero.

    Parameters
    ----------
    y_test      : pd.Series
    y_pred_test : np.ndarray
    """
    residuals = y_test.values - y_pred_test

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred_test, residuals, alpha=0.45, s=20, color="darkorange")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Predicted Price (€)")
    axes[0].set_ylabel("Residual (Actual − Predicted)")
    axes[0].set_title("Residuals vs Predicted Price")
    axes[0].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Residual (€)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Residuals")
    axes[1].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.suptitle("Residual Analysis — Decision Tree Regressor", fontsize=12)
    plt.tight_layout()
    _save(fig, "09_residuals.png")

    print(f"[residuals] Mean : €{residuals.mean():,.0f}  "
          f"(ideal = 0)  |  Std : €{residuals.std():,.0f}\n")


def plot_feature_importance(model: DecisionTreeRegressor,
                             features: list) -> None:
    """
    Horizontal bar chart of the top-15 most important features as assigned
    by the trained Decision Tree (based on weighted impurity decrease).

    Parameters
    ----------
    model    : fitted DecisionTreeRegressor
    features : list of feature names
    """
    importances = (
        pd.Series(model.feature_importances_, index=features)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    importances.head(15).plot.barh(ax=ax, color="#2980B9", edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (weighted impurity reduction)")
    ax.set_title("Top 15 Most Important Features — Decision Tree Regressor",
                 fontsize=12)
    plt.tight_layout()
    _save(fig, "10_feature_importance.png")

    print("Top 10 feature importances:")
    print(importances.head(10).to_string())
    print()


def plot_decision_tree(model: DecisionTreeRegressor, features: list) -> None:
    """
    Visualize the first 3 levels of the trained Decision Tree.

    Deeper levels are hidden to keep the chart readable; the text rules
    (print_decision_rules) show the full tree in text form.

    Parameters
    ----------
    model    : fitted DecisionTreeRegressor
    features : list of feature names
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        model,
        max_depth=3,
        feature_names=features,
        filled=True,
        rounded=True,
        impurity=False,
        precision=0,
        ax=ax,
        fontsize=8,
    )
    ax.set_title(
        f"Decision Tree — Top 3 of {model.get_depth()} levels",
        fontsize=13,
    )
    plt.tight_layout()
    _save(fig, "11_decision_tree_diagram.png")


def print_decision_rules(model: DecisionTreeRegressor,
                          features: list,
                          max_depth: int = 4) -> None:
    """
    Print the full decision rules of the trained tree in human-readable form.

    The output uses indentation to reflect nesting depth, making it easy to
    trace a specific car specification down to its price prediction leaf.

    Parameters
    ----------
    model     : fitted DecisionTreeRegressor
    features  : list of feature names
    max_depth : int — number of levels to print (default 4)
    """
    rules = export_text(model, feature_names=features, max_depth=max_depth)
    print("=" * 60)
    print(f"DECISION RULES (first {max_depth} levels)")
    print("=" * 60)
    print(rules)


# ==============================================================================
# 6. SINGLE-CAR PRICE ESTIMATOR
# ==============================================================================

def predict_price(model: DecisionTreeRegressor,
                  features: list,
                  car_specs: dict) -> float:
    """
    Predict the resale price for a single car given its specifications.

    Usage example
    -------------
    >>> specs = {
    ...     "Age_08_04": 36, "KM": 60000, "HP": 90, "Weight": 1050,
    ...     "Airco": 1, "Automatic_airco": 0, ...
    ... }
    >>> price = predict_price(model, features, specs)
    >>> print(f"Estimated price: €{price:,.0f}")

    Parameters
    ----------
    model      : fitted DecisionTreeRegressor
    features   : list of feature names (same order used during training)
    car_specs  : dict mapping feature name → value;
                 missing features default to 0

    Returns
    -------
    float
        Estimated resale price in euros
    """
    row = {f: car_specs.get(f, 0) for f in features}
    X_new = pd.DataFrame([row])[features]
    price = float(model.predict(X_new)[0])
    print(f"[predict] Estimated resale price : €{price:,.0f}")
    return price


# ==============================================================================
# MAIN — run the complete pipeline
# ==============================================================================

def main() -> None:
    """
    Execute the end-to-end analysis pipeline:
      load → clean → EDA → normalize → visualize → train → evaluate → demo

    All plots are saved to ./outputs/
    """
    print("\n" + "=" * 60)
    print("  Toyota Corolla Resale Price Analysis  (v2)")
    print("=" * 60 + "\n")

    # ── 1. Load & clean ───────────────────────────────────────────────────────
    df = load_and_clean(CSV_PATH)

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    descriptive_stats(df)
    price_corr = correlation_analysis(df)

    # ── 3. Normalize ──────────────────────────────────────────────────────────
    normalized = normalize(df)          # pd.DataFrame.normalized

    # ── 4. Visualizations ─────────────────────────────────────────────────────
    print("[plots] Generating visualizations …")
    plot_correlation_heatmap(df)
    plot_top_correlations(price_corr)
    plot_scatter_raw(df, price_corr)
    plot_scatter_normalized(normalized, price_corr, df)
    plot_boxplots_binary(df, price_corr)
    plot_boxplots_age_groups(df)

    # ── 5. Decision Tree ──────────────────────────────────────────────────────
    model, X_train, X_test, y_train, y_test, features = train_model(df)
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    plot_predicted_vs_actual(y_test, results["y_pred_test"])
    plot_residuals(y_test, results["y_pred_test"])
    plot_feature_importance(model, features)
    plot_decision_tree(model, features)
    print_decision_rules(model, features)

    # ── 6. Demo prediction ────────────────────────────────────────────────────
    demo_car = {
        "Age_08_04": 36, "Mfg_Month": 5, "Mfg_Year": 2001,
        "KM": 60000, "HP": 90, "Met_Color": 1, "Automatic": 0,
        "cc": 1600, "Doors": 3, "Cylinders": 4, "Gears": 5,
        "Quarterly_Tax": 85, "Weight": 1050, "Mfr_Guarantee": 0,
        "BOVAG_Guarantee": 1, "Guarantee_Period": 3, "ABS": 1,
        "Airbag_1": 1, "Airbag_2": 1, "Airco": 1, "Automatic_airco": 0,
        "Boardcomputer": 0, "CD_Player": 1, "Central_Lock": 1,
        "Powered_Windows": 0, "Power_Steering": 1, "Radio": 1,
        "Mistlamps": 0, "Sport_Model": 0, "Backseat_Divider": 1,
        "Metallic_Rim": 0, "Radio_cassette": 0, "Tow_Bar": 0,
    }
    print("\n[demo]  Sample car specifications:")
    print(f"        Age={demo_car['Age_08_04']} months  |  "
          f"KM={demo_car['KM']:,}  |  HP={demo_car['HP']}  |  "
          f"Airco={demo_car['Airco']}  |  Weight={demo_car['Weight']} kg")
    predict_price(model, features, demo_car)

    print(f"\n[done]  All outputs saved to → {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
