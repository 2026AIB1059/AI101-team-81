"""Automated tests for the F1 Podium Prediction project.
"""Test code author: Naitik Goyal : execution lead"""
Run from the repository root with either:
    python tests/test_project.py          (formatted report)
    python -m unittest discover -s tests -v

The tests use a small SYNTHETIC dataset with the same file layout as the Kaggle
data, written to a temporary folder. They never touch your real dataset or your
real outputs/ folder, and need no extra libraries.
"""

import contextlib
import io
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import main as main_module  # noqa: E402
import model as mdl  # noqa: E402
import plots  # noqa: E402
import utils  # noqa: E402

EXPECTED_PLOTS = [
    "podium_rate_by_grid.png",
    "top_drivers_podiums.png",
    "top_constructors_podiums.png",
    "grid_dependence_index.png",
    "confusion_matrix.png",
    "roc_curve.png",
    "calibration_curve.png",
    "feature_coefficients.png",
    "backtest_top3.png",
]


# ---------------------------------------------------------------------------
# Synthetic data in the same format as the Kaggle files
# ---------------------------------------------------------------------------
def make_synthetic_dataset(folder: Path) -> None:
    rng = np.random.default_rng(42)
    n_circuits, n_drivers, n_teams = 8, 40, 8

    pd.DataFrame(
        {
            "circuitId": range(1, n_circuits + 1),
            "name": [f"Circuit {i}" for i in range(1, n_circuits + 1)],
            "lat": rng.uniform(-30, 60, n_circuits),
            "lng": rng.uniform(-100, 140, n_circuits),
            "alt": ["\\N" if i == 3 else int(rng.integers(0, 800)) for i in range(n_circuits)],
        }
    ).to_csv(folder / "circuits.csv", index=False)

    dob = pd.to_datetime("1965-01-01") + pd.to_timedelta(rng.integers(0, 9000, n_drivers), unit="D")
    pd.DataFrame(
        {
            "driverId": range(1, n_drivers + 1),
            "forename": [f"Driver{i}" for i in range(1, n_drivers + 1)],
            "surname": [f"Surname{i}" for i in range(1, n_drivers + 1)],
            "dob": dob.strftime("%Y-%m-%d"),
        }
    ).to_csv(folder / "drivers.csv", index=False)

    pd.DataFrame(
        {
            "constructorId": range(1, n_teams + 1),
            "name": [f"Team {i}" for i in range(1, n_teams + 1)],
        }
    ).to_csv(folder / "constructors.csv", index=False)

    skill = rng.normal(0, 1, n_drivers)
    team_of = np.repeat(np.arange(1, n_teams + 1), n_drivers // n_teams)
    team_skill = rng.normal(0, 1, n_teams)
    races, results = [], []
    race_id = result_id = 1
    for year in range(1998, 2025):
        for rnd in range(1, 11):
            circuit = int(rng.integers(1, n_circuits + 1))
            races.append((race_id, year, rnd, circuit, f"Grand Prix {circuit}",
                          f"{year}-{rnd:02d}-15"))
            drivers = rng.choice(n_drivers, 20, replace=False)
            strength = skill[drivers] + team_skill[team_of[drivers] - 1]
            qualifying = strength + rng.normal(0, 1, 20)
            grid = np.empty(20, int)
            grid[np.argsort(-qualifying)] = np.arange(1, 21)
            grid[rng.integers(0, 20)] = 0  # one pit-lane start per race
            finish = strength + rng.normal(0, 1.5, 20) - 0.15 * np.where(grid == 0, 20, grid)
            position = np.empty(20, int)
            position[np.argsort(-finish)] = np.arange(1, 21)
            for k, d in enumerate(drivers):
                results.append((result_id, race_id, int(d + 1), int(team_of[d]),
                                int(grid[k]), int(position[k])))
                result_id += 1
            race_id += 1

    pd.DataFrame(races, columns=["raceId", "year", "round", "circuitId", "name", "date"]
                 ).to_csv(folder / "races.csv", index=False)
    pd.DataFrame(results, columns=["resultId", "raceId", "driverId", "constructorId",
                                   "grid", "positionOrder"]
                 ).to_csv(folder / "results.csv", index=False)


class ProjectTestCase(unittest.TestCase):
    """Builds the synthetic dataset once and points the project at it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="f1_test_"))
        cls.data_dir = cls.tmp / "Formula 1 Dataset"
        cls.out_dir = cls.tmp / "outputs"
        cls.data_dir.mkdir()
        make_synthetic_dataset(cls.data_dir)

        cls._saved = (utils.DATA_DIR, utils.OUTPUT_DIR, plots.OUTPUT_DIR)
        utils.DATA_DIR = cls.data_dir
        utils.OUTPUT_DIR = cls.out_dir
        plots.OUTPUT_DIR = cls.out_dir

        cls.raw = utils.load_data()
        cls.df = utils.build_features(cls.raw)
        cls.train, cls.val, cls.test = utils.time_split(cls.df)
        cls.circuits = mdl.circuit_dependence(cls.train)
        cls.train = mdl.add_circuit_features(cls.train, cls.circuits)
        cls.val = mdl.add_circuit_features(cls.val, cls.circuits)
        cls.test = mdl.add_circuit_features(cls.test, cls.circuits)
        cls.model = mdl.train(cls.train)

    @classmethod
    def tearDownClass(cls):
        utils.DATA_DIR, utils.OUTPUT_DIR, plots.OUTPUT_DIR = cls._saved
        shutil.rmtree(cls.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------
class TestUtils(ProjectTestCase):
    def test_load_data_has_expected_columns(self):
        """load_data joins all tables and keeps the columns we need"""
        for col in ["raceId", "driverId", "grid", "positionOrder", "year", "race_name",
                    "driver_name", "constructor_name", "circuit_name", "lat", "lng", "alt"]:
            self.assertIn(col, self.raw.columns)
        self.assertEqual(len(self.raw), 27 * 10 * 20)

    def test_missing_file_gives_clear_error(self):
        """read_csv raises FileNotFoundError that names the missing file"""
        with self.assertRaises(FileNotFoundError) as ctx:
            utils.read_csv("does_not_exist.csv")
        self.assertIn("does_not_exist.csv", str(ctx.exception))

    def test_podium_target_matches_finishing_position(self):
        """podium = 1 exactly when positionOrder <= 3 (3 per race)"""
        self.assertTrue(((self.df["positionOrder"] <= 3) == (self.df["podium"] == 1)).all())
        self.assertTrue((self.df.groupby("raceId")["podium"].sum() == 3).all())

    def test_pit_lane_starts_are_recoded(self):
        """grid = 0 never survives feature engineering"""
        self.assertTrue((self.raw["grid"] == 0).any())
        self.assertFalse((self.df["grid"] == 0).any())

    def test_missing_circuit_altitude_is_filled(self):
        """'\\N' altitude values are read as missing and then imputed"""
        self.assertTrue(self.raw["alt"].isna().any())
        self.assertFalse(self.df["circuit_alt"].isna().any())

    def test_no_missing_values_in_features(self):
        """every model feature is present and has no NaN"""
        for col in utils.FEATURES[:1] + [f for f in utils.FEATURES if f not in
                                         ("circuit_dependence", "grid_x_circuit")]:
            self.assertIn(col, self.df.columns)
            self.assertFalse(self.df[col].isna().any(), msg=col)

    def test_first_race_features_use_defaults_only(self):
        """no leakage: a driver's first race has experience 0 and default form"""
        first = self.df.sort_values(["date", "raceId"]).groupby("driverId").head(1)
        self.assertTrue((first["driver_experience"] == 0).all())
        self.assertTrue(np.allclose(first["driver_form"], utils.DEFAULT_DRIVER_FORM))

    def test_driver_form_uses_only_past_races(self):
        """no leakage: form in race N equals the finishing fraction of race N-1"""
        driver = self.df["driverId"].value_counts().index[0]
        rows = self.df[self.df["driverId"] == driver].sort_values(["date", "raceId"])
        self.assertAlmostEqual(rows["driver_form"].iloc[1], rows["finish_pct"].iloc[0])

    def test_constructor_form_starts_at_zero(self):
        """no leakage: a team's first ever race has constructor_form 0"""
        first = self.df.sort_values(["date", "raceId"]).groupby("constructorId").head(1)
        self.assertTrue((first["constructor_form"] == 0).all())

    def test_time_split_is_chronological_and_disjoint(self):
        """train < validation < test in time, and the sets do not overlap"""
        self.assertLessEqual(self.train["year"].max(), utils.TRAIN_END)
        self.assertTrue((self.val["year"] == utils.VAL_YEAR).all())
        self.assertGreaterEqual(self.test["year"].min(), utils.TEST_START)
        ids = [set(d.index) for d in (self.train, self.val, self.test)]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_time_split_empty_raises(self):
        """time_split raises a ValueError if a split would be empty"""
        with self.assertRaises(ValueError):
            utils.time_split(self.df[self.df["year"] < 2010])

    def test_format_metrics_is_aligned_text(self):
        """format_metrics returns one readable line"""
        line = utils.format_metrics("LogReg (test)", {"Accuracy": 0.9, "ROC-AUC": 0.95})
        self.assertIn("Accuracy: 0.900", line)
        self.assertIn("ROC-AUC: 0.950", line)


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------
class TestModel(ProjectTestCase):
    def test_circuit_dependence_table(self):
        """Grid Dependence Index: one row per circuit, sorted high to low"""
        table = self.circuits
        self.assertGreater(len(table), 0)
        self.assertTrue(table["dependence_index"].is_monotonic_decreasing)
        for col in ["circuit_name", "races", "pole_win_rate", "pole_podium_rate"]:
            self.assertIn(col, table.columns)
        self.assertTrue(table["pole_win_rate"].between(0, 1).all())

    def test_circuit_dependence_skips_small_circuits(self):
        """circuits with too few races are left out of the index"""
        self.assertEqual(len(mdl.circuit_dependence(self.train, min_races=10_000)), 0)

    def test_circuit_features_have_no_gaps(self):
        """add_circuit_features gives every row an index and a grid interaction"""
        for part in (self.train, self.val, self.test):
            self.assertFalse(part[["circuit_dependence", "grid_x_circuit"]].isna().any().any())
            self.assertTrue(np.allclose(part["grid_x_circuit"],
                                        part["grid"] * part["circuit_dependence"]))

    def test_unseen_circuit_gets_average_index(self):
        """a circuit missing from the table falls back to the mean index"""
        row = self.test.head(1).copy()
        row["circuitId"] = 9999
        out = mdl.add_circuit_features(row, self.circuits)
        self.assertAlmostEqual(out["circuit_dependence"].iloc[0],
                               self.circuits["dependence_index"].mean())

    def test_model_probabilities_are_valid(self):
        """predict_proba returns values between 0 and 1"""
        prob = self.model.predict_proba(self.test[utils.FEATURES])[:, 1]
        self.assertEqual(len(prob), len(self.test))
        self.assertTrue(((prob >= 0) & (prob <= 1)).all())

    def test_model_beats_random_guessing(self):
        """test ROC-AUC is clearly above 0.5 (chance level)"""
        self.assertGreater(mdl.evaluate_model(self.model, self.test)["ROC-AUC"], 0.7)

    def test_better_grid_slot_means_higher_podium_probability(self):
        """moving the same driver from P20 to P1 raises the podium probability"""
        base = self.test.head(1).copy()
        front, back = base.copy(), base.copy()
        front["grid"], back["grid"] = 1, 20
        front = mdl.add_circuit_features(front, self.circuits)
        back = mdl.add_circuit_features(back, self.circuits)
        p_front = self.model.predict_proba(front[utils.FEATURES])[0, 1]
        p_back = self.model.predict_proba(back[utils.FEATURES])[0, 1]
        self.assertGreater(p_front, p_back)

    def test_metrics_dictionary(self):
        """evaluate returns all six metrics in valid ranges"""
        metrics = mdl.evaluate_model(self.model, self.val)
        self.assertEqual(set(metrics), {"Accuracy", "Precision", "Recall",
                                        "ROC-AUC", "LogLoss", "Brier"})
        for name in ("Accuracy", "Precision", "Recall", "ROC-AUC", "Brier"):
            self.assertTrue(0 <= metrics[name] <= 1, msg=name)
        self.assertGreater(metrics["LogLoss"], 0)

    def test_baseline_metrics(self):
        """the 'grid <= 3' baseline produces the same metric set"""
        base = mdl.evaluate_baseline(self.test)
        self.assertEqual(set(base), set(mdl.evaluate_model(self.model, self.test)))
        self.assertGreater(base["Recall"], 0)

    def test_coefficient_table(self):
        """one row per feature, odds ratio = exp(coefficient)"""
        table = mdl.coefficient_table(self.model)
        self.assertEqual(set(table["feature"]), set(utils.FEATURES))
        self.assertTrue(np.allclose(table["odds_ratio_per_sd"], np.exp(table["coefficient"])))
        self.assertTrue(table["coefficient"].abs().is_monotonic_decreasing)

    def test_podium_leaders(self):
        """podium counts are sorted and never exceed the number of races"""
        leaders = mdl.podium_leaders(self.df, "driver_name", top=5)
        self.assertEqual(len(leaders), 5)
        self.assertTrue(leaders.is_monotonic_decreasing)
        self.assertLessEqual(leaders.iloc[0], self.df["raceId"].nunique())
        teams = mdl.podium_leaders(self.df, "constructor_name", top=3)
        self.assertEqual(len(teams), 3)

    def test_backtest_hits_are_between_0_and_3(self):
        """each race: model and baseline get 0 to 3 podium finishers right"""
        bt = mdl.backtest_top3(self.model, self.test)
        self.assertEqual(len(bt), self.test["raceId"].nunique())
        self.assertTrue(bt["model_hits"].between(0, 3).all())
        self.assertTrue(bt["baseline_hits"].between(0, 3).all())

    def test_top3_for_race(self):
        """actual and predicted top 3 each contain exactly three drivers"""
        race_id = self.test["raceId"].iloc[-1]
        actual, predicted = mdl.top3_for_race(self.model, self.test, race_id)
        self.assertEqual(len(actual), 3)
        self.assertEqual(len(predicted), 3)
        self.assertEqual(list(actual["positionOrder"]), [1, 2, 3])
        self.assertTrue(predicted["prob"].is_monotonic_decreasing)


# ---------------------------------------------------------------------------
# plots.py
# ---------------------------------------------------------------------------
class TestPlots(ProjectTestCase):
    def setUp(self):
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def assert_file(self, name):
        path = self.out_dir / name
        self.assertTrue(path.exists(), msg=f"{name} was not created")
        self.assertGreater(path.stat().st_size, 1000, msg=f"{name} looks empty")

    def run_quietly(self, func, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            func(*args)

    def test_podium_rate_plot(self):
        """podium_rate_by_grid.png is created"""
        self.run_quietly(plots.podium_rate_by_grid, self.train)
        self.assert_file("podium_rate_by_grid.png")

    def test_leader_plots(self):
        """driver and constructor leader charts are created"""
        self.run_quietly(plots.leaders_plot, mdl.podium_leaders(self.df, "driver_name"),
                         "Drivers", "top_drivers_podiums.png", plots.BLUE)
        self.run_quietly(plots.leaders_plot, mdl.podium_leaders(self.df, "constructor_name"),
                         "Teams", "top_constructors_podiums.png", plots.TEAL)
        self.assert_file("top_drivers_podiums.png")
        self.assert_file("top_constructors_podiums.png")

    def test_grid_dependence_plot_with_few_circuits(self):
        """the circuit chart also works when there are fewer than 16 circuits"""
        self.run_quietly(plots.grid_dependence_plot, self.circuits)
        self.assert_file("grid_dependence_index.png")

    def test_evaluation_plots(self):
        """confusion matrix, ROC, calibration and coefficient plots are created"""
        prob = self.model.predict_proba(self.test[utils.FEATURES])[:, 1]
        self.run_quietly(plots.confusion_matrix_plot, self.test["podium"], (prob >= 0.5).astype(int))
        self.run_quietly(plots.roc_plot, self.test["podium"], prob)
        self.run_quietly(plots.calibration_plot, self.test["podium"], prob)
        self.run_quietly(plots.coefficient_plot, mdl.coefficient_table(self.model))
        for name in ("confusion_matrix.png", "roc_curve.png",
                     "calibration_curve.png", "feature_coefficients.png"):
            self.assert_file(name)

    def test_backtest_plot(self):
        """backtest_top3.png is created"""
        self.run_quietly(plots.backtest_plot, mdl.backtest_top3(self.model, self.test))
        self.assert_file("backtest_top3.png")


# ---------------------------------------------------------------------------
# main.py (end to end)
# ---------------------------------------------------------------------------
class TestMainEndToEnd(ProjectTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        shutil.rmtree(cls.out_dir, ignore_errors=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main_module.main()
        cls.stdout = buffer.getvalue()

    def test_main_prints_every_section(self):
        """main prints leaders, circuit index, results, odds ratios, back-test"""
        for heading in [
            "F1 Podium Prediction",
            "Most podium finishes of all time",
            "Grid Dependence Index by circuit",
            "Results (Logistic Regression vs baseline",
            "Coefficients and odds ratios",
            "Back-test on",
            "Latest race in the data",
            "Saving plots",
            "Done in",
        ]:
            self.assertIn(heading, self.stdout)

    def test_main_reports_metrics_for_model_and_baseline(self):
        """results table has baseline, validation and test rows with ROC-AUC"""
        for label in ("Baseline  (test)", "LogReg    (validation)", "LogReg    (test)"):
            self.assertIn(label, self.stdout)
        self.assertGreaterEqual(self.stdout.count("ROC-AUC:"), 3)

    def test_main_names_the_top_three(self):
        """main shows the actual and the predicted top 3 of the latest race"""
        self.assertIn("Actual top 3:", self.stdout)
        self.assertIn("Model's predicted top 3:", self.stdout)
        self.assertIn("P1", self.stdout)
        self.assertIn("podium probability", self.stdout)

    def test_main_saves_all_plots(self):
        """all 9 plots exist in outputs/ after main runs"""
        for name in EXPECTED_PLOTS:
            path = self.out_dir / name
            self.assertTrue(path.exists(), msg=f"missing {name}")
            self.assertGreater(path.stat().st_size, 1000, msg=f"{name} looks empty")
        self.assertEqual(len(list(self.out_dir.glob("*.png"))), len(EXPECTED_PLOTS))


# ---------------------------------------------------------------------------
# Formatted report
# ---------------------------------------------------------------------------
class ReportResult(unittest.TextTestResult):
    """Prints one PASS/FAIL line per test, grouped by module."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self._current = None
        self._start = 0.0
        self.rows = []

    def _label(self, test):
        return (test.shortDescription() or test._testMethodName).strip()

    def startTest(self, test):
        super().startTest(test)
        group = type(test).__name__
        if group != self._current:
            self._current = group
            names = {"TestUtils": "utils.py", "TestModel": "model.py",
                     "TestPlots": "plots.py", "TestMainEndToEnd": "main.py (end to end)"}
            self.stream.write(f"\n  {names.get(group, group)}\n  " + "-" * 78 + "\n")
        self._start = time.time()

    def _row(self, test, status):
        ms = (time.time() - self._start) * 1000
        self.stream.write(f"  [{status:<5}] {self._label(test):<66}{ms:>5.0f} ms\n")
        self.stream.flush()

    def addSuccess(self, test):
        unittest.TestResult.addSuccess(self, test)
        self._row(test, "PASS")

    def addFailure(self, test, err):
        unittest.TestResult.addFailure(self, test, err)
        self._row(test, "FAIL")

    def addError(self, test, err):
        unittest.TestResult.addError(self, test, err)
        self._row(test, "ERROR")


def run_report() -> int:
    print("=" * 80)
    print(" AI101 Mini-Project | F1 Podium Prediction | Test report")
    print("=" * 80)
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(case)
        for case in (TestUtils, TestModel, TestPlots, TestMainEndToEnd)
    )
    runner = unittest.TextTestRunner(stream=sys.stdout, resultclass=ReportResult, verbosity=0)
    result = runner.run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print("\n" + "=" * 80)
    print(f" Tests run: {result.testsRun} | Passed: {passed} | "
          f"Failed: {len(result.failures)} | Errors: {len(result.errors)}")
    print(" RESULT: " + ("ALL TESTS PASSED" if result.wasSuccessful() else "SOME TESTS FAILED"))
    print("=" * 80)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_report())
