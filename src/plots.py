"""Visualisation scripts. All figures are saved in the outputs/ folder."""
""" Author Naitik Goyal : (coding/execution lead)"""
    import matplotlib

matplotlib.use("Agg")  # no display needed, works on any laptop / server
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from utils import OUTPUT_DIR

BLUE, TEAL, ORANGE = "#1a0dab", "#0b7a75", "#c4420c"


def _save(fig, name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=150)
    plt.close(fig)
    print(f"Saved outputs/{name}")


def podium_rate_by_grid(df) -> None:
    rate = df[df["grid"] <= 20].groupby("grid")["podium"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rate.index, rate.values, color=BLUE)
    ax.set_xlabel("Starting grid position")
    ax.set_ylabel("Podium rate")
    ax.set_title("Podium rate by starting grid position")
    _save(fig, "podium_rate_by_grid.png")


def leaders_plot(counts: pd.Series, title: str, filename: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(counts.index[::-1], counts.values[::-1], color=color)
    ax.set_xlabel("Podium finishes (top 3)")
    ax.set_title(title)
    _save(fig, filename)


def grid_dependence_plot(table: pd.DataFrame, n: int = 8) -> None:
    """Most grid-dependent and least grid-dependent circuits."""
    top = table.head(n)
    bottom = table.tail(n)
    shown = pd.concat([top, bottom]).drop_duplicates("circuitId")
    colors = [ORANGE if cid in set(top["circuitId"]) else TEAL for cid in shown["circuitId"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shown["circuit_name"][::-1], shown["dependence_index"][::-1], color=colors[::-1])
    ax.set_xlabel("Grid Dependence Index (higher = grid slot matters more)")
    ax.set_title("Is the race won on Saturday? Circuits ranked by grid dependence")
    _save(fig, "grid_dependence_index.png")


def confusion_matrix_plot(y_true, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No podium", "Podium"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion matrix (test set)")
    _save(fig, "confusion_matrix.png")


def roc_plot(y_true, y_prob) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, name="Logistic Regression", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title("ROC curve (test set)")
    ax.legend()
    _save(fig, "roc_curve.png")


def calibration_plot(y_true, y_prob) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_pred, frac_pos, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Predicted podium probability")
    ax.set_ylabel("Actual podium frequency")
    ax.set_title("Calibration curve (test set)")
    ax.legend()
    _save(fig, "calibration_curve.png")


def coefficient_plot(table: pd.DataFrame) -> None:
    t = table.iloc[::-1]
    colors = [ORANGE if v < 0 else TEAL for v in t["coefficient"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(t["feature"], t["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardised coefficient (effect on podium log-odds)")
    ax.set_title("What drives a podium finish?")
    _save(fig, "feature_coefficients.png")


def backtest_plot(backtest: pd.DataFrame) -> None:
    """Average correct podium picks per race, per season: model vs baseline."""
    by_year = backtest.groupby("year")[["model_hits", "baseline_hits"]].mean()
    x = range(len(by_year))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - 0.2 for i in x], by_year["model_hits"], width=0.4, label="Logistic Regression", color=BLUE)
    ax.bar([i + 0.2 for i in x], by_year["baseline_hits"], width=0.4, label="Baseline (grid <= 3)", color="#888888")
    ax.set_xticks(list(x))
    ax.set_xticklabels(by_year.index)
    ax.set_ylim(0, 3)
    ax.set_ylabel("Correct podium picks per race (max 3)")
    ax.set_title("Back-test: predicting the top three")
    ax.legend()
    _save(fig, "backtest_top3.png")

