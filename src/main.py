"""F1 Podium Prediction: main entry point.
"""Author Naitik Goyal : Execution lead : AI modelling
Run from the repository root:
    python src/main.py

"""

import time

import plots
import model as mdl
import utils


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * 70)


def main() -> None:
    start = time.time()
    print("=" * 70)
    print(" AI101 Mini-Project | F1 Podium Prediction (Team 81)")
    print("=" * 70)

    # 1. Load the dataset and build features -------------------------------
    raw = utils.load_data()
    df = utils.build_features(raw)
    print(
        f"Loaded {len(df)} race results, seasons {int(df['year'].min())}-{int(df['year'].max())}, "
        f"{df['raceId'].nunique()} races, {df['circuitId'].nunique()} circuits"
    )

        # 2. (Who finishes in the top three? ------------------------------------
    section("Most podium finishes of all time (top 3)")
    drivers = mdl.podium_leaders(df, "driver_name", top=10)
    teams = mdl.podium_leaders(df, "constructor_name", top=10)
    print("Drivers:")
    for name, n in drivers.items():
        print(f"  {name:<28}{int(n):>5}")
    print("Constructors:")
    for name, n in teams.items():
        print(f"  {name:<28}{int(n):>5}")

    # 3. Time-based split and circuit analysis -----------------------------
    train, val, test = utils.time_split(df)
    print(
        f"\nTrain: {len(train)} rows ({utils.FIRST_SEASON}-{utils.TRAIN_END}) | "
        f"Validation: {len(val)} rows ({utils.VAL_YEAR}) | "
        f"Test: {len(test)} rows ({utils.TEST_START}+)"
    )

    section("Is the race won on Saturday? Grid Dependence Index by circuit")
    all_circuits = mdl.circuit_dependence(df)
    print(f"Circuits with at least {utils.MIN_CIRCUIT_RACES} races: {len(all_circuits)}")
    print("Most grid-dependent (grid is destiny):")
    for _, r in all_circuits.head(5).iterrows():
        print(f"  {r['circuit_name']:<34} index {r['dependence_index']:.3f}  "
              f"pole wins {r['pole_win_rate']:.0%}")
    print("Least grid-dependent (anything can happen):")
    for _, r in all_circuits.tail(5).iloc[::-1].iterrows():
        print(f"  {r['circuit_name']:<34} index {r['dependence_index']:.3f}  "
              f"pole wins {r['pole_win_rate']:.0%}")

    # The model only sees circuit effects learned from TRAINING seasons
    train_circuits = mdl.circuit_dependence(train)
    train = mdl.add_circuit_features(train, train_circuits)
    val = mdl.add_circuit_features(val, train_circuits)
    test = mdl.add_circuit_features(test, train_circuits)

    # 4. Train Logistic Regression ----------------------------------------
    model = mdl.train(train)

    section("Results (Logistic Regression vs baseline 'podium if grid <= 3')")
    print(utils.format_metrics("Baseline  (test)", mdl.evaluate_baseline(test)))
    print(utils.format_metrics("LogReg    (validation)", mdl.evaluate_model(model, val)))
    print(utils.format_metrics("LogReg    (test)", mdl.evaluate_model(model, test)))

    # 5. Interpretation: coefficients and odds ratios -----------------------
    section("Coefficients and odds ratios (largest effect first)")
    coefs = mdl.coefficient_table(model)
    print(f"{'feature':<22}{'coef':>9}{'odds/SD':>10}{'odds/unit':>11}")
    for _, r in coefs.iterrows():
        print(f"{r['feature']:<22}{r['coefficient']:>+9.3f}"
              f"{r['odds_ratio_per_sd']:>10.3f}{r['odds_ratio_per_unit']:>11.3f}")
    print("Note: read 'grid' and 'grid_x_circuit' together. The effect of one extra grid")
    print("slot is coef(grid) + coef(grid_x_circuit) x the circuit's dependence index.")

    # 6. Back-test: can we pick the top three of each race? -----------------
    section(f"Back-test on {utils.TEST_START}+ races: picking the top three")
    bt = mdl.backtest_top3(model, test)
    print(f"Races: {len(bt)}")
    print(f"Model    : {bt['model_hits'].mean():.2f} of 3 podium finishers correct per race, "
          f"exact podium set in {(bt['model_hits'] == 3).mean():.0%} of races")
    print(f"Baseline : {bt['baseline_hits'].mean():.2f} of 3 podium finishers correct per race, "
          f"exact podium set in {(bt['baseline_hits'] == 3).mean():.0%} of races")

    last_race = test.sort_values(["date", "raceId"])["raceId"].iloc[-1]
    actual, predicted = mdl.top3_for_race(model, test, last_race)
    race = test[test["raceId"] == last_race].iloc[0]
    print(f"\nLatest race in the data: {int(race['year'])} {race['race_name']}")
    print("Actual top 3:")
    for _, r in actual.iterrows():
        print(f"  P{int(r['positionOrder'])}  {r['driver_name']:<24}{r['constructor_name']:<18}"
              f"(started P{int(r['grid'])})")
    print("Model's predicted top 3:")
    for _, r in predicted.iterrows():
        print(f"      {r['driver_name']:<24}{r['constructor_name']:<18}"
              f"podium probability {r['prob']:.0%}")

    # 7. Plots -------------------------------------------------------------
    section("Saving plots")
    prob = model.predict_proba(test[utils.FEATURES])[:, 1]
    pred = (prob >= 0.5).astype(int)
    plots.podium_rate_by_grid(train)
    plots.leaders_plot(drivers, "Most podium finishes: drivers", "top_drivers_podiums.png", plots.BLUE)
    plots.leaders_plot(teams, "Most podium finishes: constructors", "top_constructors_podiums.png", plots.TEAL)
    plots.grid_dependence_plot(all_circuits)
    plots.confusion_matrix_plot(test["podium"], pred)
    plots.roc_plot(test["podium"], prob)
    plots.calibration_plot(test["podium"], prob)
    plots.coefficient_plot(coefs)
    plots.backtest_plot(bt)

    print(f"\nDone in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
