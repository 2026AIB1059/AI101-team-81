"""Model building, training, evaluation, circuit analysis and the podium back-test."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import FEATURES, MIN_CIRCUIT_RACES


# ---------------------------------------------------------------------------
# Circuit analysis: how much does the grid slot decide the result?
# ---------------------------------------------------------------------------
def circuit_dependence(df: pd.DataFrame, min_races: int = MIN_CIRCUIT_RACES) -> pd.DataFrame:
    """Grid Dependence Index per circuit.

    For each circuit we fit podium ~ grid. The index is minus the slope, so a
    HIGHER value means the grid slot matters MORE ("grid is destiny"), and a
    value near 0 means "anything can happen".
    """
    rows = []
    for circuit_id, g in df.groupby("circuitId"):
        n_races = g["raceId"].nunique()
        if n_races < min_races or g["podium"].nunique() < 2:
            continue
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(g[["grid"]], g["podium"])
        pole = g[g["grid"] == 1]
        rows.append(
            {
                "circuitId": circuit_id,
                "circuit_name": g["circuit_name"].iloc[0],
                "races": n_races,
                "dependence_index": -lr.coef_[0][0],
                "pole_win_rate": (pole["positionOrder"] == 1).mean() if len(pole) else np.nan,
                "pole_podium_rate": pole["podium"].mean() if len(pole) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("dependence_index", ascending=False).reset_index(drop=True)


def add_circuit_features(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Attach the circuit index (learned on TRAIN data only) and the grid interaction."""
    df = df.copy()
    default = table["dependence_index"].mean()
    index_map = table.set_index("circuitId")["dependence_index"]
    df["circuit_dependence"] = df["circuitId"].map(index_map).fillna(default)
    df["grid_x_circuit"] = df["grid"] * df["circuit_dependence"]
    return df


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------
def build_model():
    """Scaled Logistic Regression (scaling makes coefficients comparable)."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def train(train_df: pd.DataFrame):
    model = build_model()
    model.fit(train_df[FEATURES], train_df["podium"])
    return model


def evaluate(y_true, y_pred, y_prob) -> dict:
    """Classification and probability metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "LogLoss": log_loss(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
    }


def evaluate_model(model, df: pd.DataFrame) -> dict:
    prob = model.predict_proba(df[FEATURES])[:, 1]
    return evaluate(df["podium"], (prob >= 0.5).astype(int), prob)


def evaluate_baseline(df: pd.DataFrame) -> dict:
    """Naive baseline: predict a podium if the driver starts in the top 3."""
    pred = (df["grid"] <= 3).astype(int)
    prob = np.clip(1.0 / df["grid"], 0.001, 0.999)  # smooth score for AUC / log loss
    return evaluate(df["podium"], pred, prob)


def coefficient_table(model) -> pd.DataFrame:
    """Standardised coefficients and odds ratios, largest effect first."""
    scaler = model.named_steps["standardscaler"]
    coef = model.named_steps["logisticregression"].coef_[0]
    table = pd.DataFrame(
        {
            "feature": FEATURES,
            "coefficient": coef,
            "odds_ratio_per_sd": np.exp(coef),
            "odds_ratio_per_unit": np.exp(coef / scaler.scale_),
        }
    )
    order = table["coefficient"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Who finishes in the top three? Leaders and back-test
# ---------------------------------------------------------------------------
def podium_leaders(df: pd.DataFrame, by: str, top: int = 10) -> pd.Series:
    """All-time podium counts by 'driver_name' or 'constructor_name'."""
    return df.groupby(by)["podium"].sum().sort_values(ascending=False).head(top)


def backtest_top3(model, df: pd.DataFrame) -> pd.DataFrame:
    """For every race, pick the 3 drivers with the highest predicted podium
    probability and count how many really finished in the top 3 (0 to 3).
    The baseline picks the three drivers on the front of the grid."""
    df = df.copy()
    df["prob"] = model.predict_proba(df[FEATURES])[:, 1]
    rows = []
    for race_id, g in df.groupby("raceId"):
        picked = g.nlargest(3, "prob")
        front = g.nsmallest(3, "grid")
        rows.append(
            {
                "raceId": race_id,
                "year": int(g["year"].iloc[0]),
                "race_name": g["race_name"].iloc[0],
                "model_hits": int(picked["podium"].sum()),
                "baseline_hits": int(front["podium"].sum()),
            }
        )
    return pd.DataFrame(rows)


def top3_for_race(model, df: pd.DataFrame, race_id: int):
    """Return (actual top 3, predicted top 3) for one race."""
    g = df[df["raceId"] == race_id].copy()
    g["prob"] = model.predict_proba(g[FEATURES])[:, 1]
    cols = ["driver_name", "constructor_name", "grid", "positionOrder", "prob"]
    actual = g.nsmallest(3, "positionOrder")[cols]
    predicted = g.nlargest(3, "prob")[cols]
    return actual, predicted
