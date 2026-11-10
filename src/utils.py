"""Helper functions: paths, settings, data loading, feature engineering, formatting."""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (always relative to the repository root, never absolute)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Formula 1 Dataset"
OUTPUT_DIR = ROOT / "outputs"

# ---------------------------------------------------------------------------
# Project settings
# ---------------------------------------------------------------------------
FIRST_SEASON = 1950          # use the whole dataset (1950-2024)
TRAIN_END = 2021             # train: FIRST_SEASON .. 2021
VAL_YEAR = 2022              # validation: 2022
TEST_START = 2023            # test (back-test): 2023 .. 2024
FORM_WINDOW = 5              # rolling window (races) for "form" features
MIN_CIRCUIT_RACES = 10       # circuits with fewer races get the average effect
DEFAULT_DRIVER_FORM = 0.6    # fill value for a driver's very first race

FEATURES = [
    "grid",                  # starting grid position
    "circuit_dependence",    # how much this circuit rewards a good grid slot
    "grid_x_circuit",        # grid x circuit dependence (the interaction)
    "driver_experience",     # number of earlier races
    "driver_age",
    "driver_form",           # recent finishing position (0 = win, 1 = last)
    "constructor_form",      # recent podiums per race of the team
    "circuit_lat",
    "circuit_lng",
    "circuit_alt",
]


def read_csv(name: str) -> pd.DataFrame:
    """Read one CSV from the dataset folder ('\\N' marks missing values)."""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{name}' in '{DATA_DIR}'. "
            "Check that the 'Formula 1 Dataset' folder is in the repository root."
        )
    return pd.read_csv(path, na_values=["\\N"])


def load_data() -> pd.DataFrame:
    """Join results, races, drivers, constructors and circuits into one table."""
    results = read_csv("results.csv")[
        ["raceId", "driverId", "constructorId", "grid", "positionOrder"]
    ]
    races = read_csv("races.csv")[
        ["raceId", "year", "round", "circuitId", "name", "date"]
    ].rename(columns={"name": "race_name"})
    drivers = read_csv("drivers.csv")[["driverId", "forename", "surname", "dob"]]
    constructors = read_csv("constructors.csv")[["constructorId", "name"]].rename(
        columns={"name": "constructor_name"}
    )
    circuits = read_csv("circuits.csv")[["circuitId", "name", "lat", "lng", "alt"]].rename(
        columns={"name": "circuit_name"}
    )

    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]

    df = (
        results.merge(races, on="raceId")
        .merge(drivers[["driverId", "driver_name", "dob"]], on="driverId")
        .merge(constructors, on="constructorId")
        .merge(circuits, on="circuitId")
    )
    df["date"] = pd.to_datetime(df["date"])
    df["dob"] = pd.to_datetime(df["dob"])
    return df.sort_values(["date", "raceId"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the target and the features.

    Every rolling feature is shifted by one race, so a row only uses
    information from races that happened BEFORE it (no data leakage).
    """
    df = df.sort_values(["date", "raceId"]).copy()

    # Target: podium finish (top 3)
    df["podium"] = (df["positionOrder"] <= 3).astype(int)

    # Pit-lane starts are stored as grid = 0 -> treat as last on the grid
    max_grid = df.groupby("raceId")["grid"].transform("max")
    df.loc[df["grid"] == 0, "grid"] = max_grid + 1

    # Finishing position as a fraction of the field (fair across eras)
    field_size = df.groupby("raceId")["positionOrder"].transform("count")
    df["finish_pct"] = (df["positionOrder"] - 1) / (field_size - 1).clip(lower=1)

    # Driver experience: number of earlier races
    df["driver_experience"] = df.groupby("driverId").cumcount()

    # Driver age (years) on race day
    df["driver_age"] = (df["date"] - df["dob"]).dt.days / 365.25
    df["driver_age"] = df["driver_age"].fillna(df["driver_age"].median())

    # Driver form: mean finishing fraction over the previous races
    df["driver_form"] = (
        df.groupby("driverId")["finish_pct"]
        .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
        .fillna(DEFAULT_DRIVER_FORM)
    )

    # Constructor form: mean podiums per race of the team over previous races
    team = (
        df.groupby(["constructorId", "raceId", "date"], as_index=False)["podium"]
        .sum()
        .rename(columns={"podium": "team_podiums"})
        .sort_values(["date", "raceId"])
    )
    team["constructor_form"] = (
        team.groupby("constructorId")["team_podiums"]
        .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
        .fillna(0.0)
    )
    df = df.merge(
        team[["constructorId", "raceId", "constructor_form"]],
        on=["constructorId", "raceId"],
        how="left",
    )

    # Circuit attributes
    df = df.rename(columns={"lat": "circuit_lat", "lng": "circuit_lng", "alt": "circuit_alt"})
    df["circuit_alt"] = df["circuit_alt"].fillna(df["circuit_alt"].median())
    df["circuit_lat"] = df["circuit_lat"].fillna(df["circuit_lat"].median())
    df["circuit_lng"] = df["circuit_lng"].fillna(df["circuit_lng"].median())

    df = df[df["year"] >= FIRST_SEASON]
    return df.sort_values(["date", "raceId"]).reset_index(drop=True)


def time_split(df: pd.DataFrame):
    """Time-based split (a random split would leak the future into training)."""
    train = df[df["year"] <= TRAIN_END]
    val = df[df["year"] == VAL_YEAR]
    test = df[df["year"] >= TEST_START]
    if train.empty or val.empty or test.empty:
        raise ValueError(
            "Train/validation/test split is empty. "
            "Check the season ranges in src/utils.py and the dataset files."
        )
    return train.copy(), val.copy(), test.copy()


def format_metrics(name: str, metrics: dict) -> str:
    """Return one aligned line of metrics for printing."""
    parts = [f"{key}: {value:.3f}" for key, value in metrics.items()]
    return f"{name:<24}" + " | ".join(parts)
