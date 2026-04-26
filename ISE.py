from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

#Constants
SYSTEMS = ["batlik", "dconvert", "h2", "jump3r", "kanzi", "lrzip", "x264", "xz", "z3"]
METRICS = ["mape", "mae", "rmse"]
DATASET_DIR = "datasets"
RESULTS_DIR = "results"
TRAIN_FRAC = 0.7
VAL_FRAC = 0.2
REPEATS = 30
SEED = 42


def scoring(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"mape": mape, "mae": mae, "rmse": rmse}


def split_cols(X):
    # which columns are categorical vs numeric
    num_cols = []
    cat_cols = []

    for col in X.columns:
        vals = X[col].dropna()
        is_text = pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col])
        is_int = pd.api.types.is_integer_dtype(X[col])
        few_unique = vals.nunique() <= 10

        if is_text or (is_int and few_unique):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    return num_cols, cat_cols


def prep(num_cols, cat_cols, scale=False):
    parts = []

    if num_cols:
        num_steps = [("fill", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("scale", StandardScaler()))
        parts.append(("numbers", Pipeline(num_steps), num_cols))

    if cat_cols:
        cat_steps = [
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
        parts.append(("categories", Pipeline(cat_steps), cat_cols))

    return ColumnTransformer(parts, remainder="drop")


def pick_model(X_train, y_train, num_cols, cat_cols, seed):
    # validation set to choose between ridge and cart
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VAL_FRAC, random_state=seed, shuffle=True
    )

    all_cols = list(X_train.columns)

    ridge = Pipeline([
        ("prep", prep(num_cols, cat_cols, scale=True)),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 9))),
    ])

    cart = Pipeline([
        ("prep", prep(all_cols, [], scale=False)),
        ("model", DecisionTreeRegressor(max_depth=10, min_samples_leaf=3, random_state=0)),
    ])

    candidates = [("ridge_onehot", ridge), ("cart", cart)]

    best_name, best_model = candidates[0]
    best_model.fit(X_fit, y_fit)
    best_mape = scoring(y_val, best_model.predict(X_val))["mape"]

    for name, model in candidates[1:]:
        model.fit(X_fit, y_fit)
        preds = model.predict(X_val)
        val_mape = scoring(y_val, preds)["mape"]

        if val_mape < best_mape:
            best_mape = val_mape
            best_name = name
            best_model = model

    best_model.fit(X_train, y_train)
    return best_name, best_model


def run_set(system, csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].copy()
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce")

    num_cols, cat_cols = split_cols(X)
    rows = []

    for repeat in range(REPEATS):
        seed = SEED + repeat

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=TRAIN_FRAC, random_state=seed, shuffle=True
        )

        # baseline: lab2 baseline linear regression
        baseline_model = LinearRegression()
        baseline_model.fit(X_train, y_train)
        baseline_preds = baseline_model.predict(X_test)
        baseline_scores = scoring(y_test, baseline_preds)

        chosen_name, chosen_model = pick_model(X_train, y_train, num_cols, cat_cols, seed)
        improved_preds = chosen_model.predict(X_test)
        improved_scores = scoring(y_test, improved_preds)

        rows.append({
            "approach": "baseline",
            "chosen_model": "linear_regression",
            "system": system,
            "dataset": csv_path.name,
            "repeat": repeat + 1,
            "num_rows": len(X),
            "num_features": X.shape[1],
            "mape": baseline_scores["mape"],
            "mae": baseline_scores["mae"],
            "rmse": baseline_scores["rmse"],
        })

        rows.append({
            "approach": "improved",
            "chosen_model": chosen_name,
            "system": system,
            "dataset": csv_path.name,
            "repeat": repeat + 1,
            "num_rows": len(X),
            "num_features": X.shape[1],
            "mape": improved_scores["mape"],
            "mae": improved_scores["mae"],
            "rmse": improved_scores["rmse"],
        })

    return rows


def make_summaries(raw):
    dataset_rows = []

    for (approach, system, dataset), group in raw.groupby(["approach", "system", "dataset"]):
        row = {
            "approach": approach,
            "system": system,
            "dataset": dataset,
            "repeats": len(group),
            "most_selected_model": group["chosen_model"].value_counts().index[0],
        }
        for m in METRICS:
            row[f"{m}_mean"] = group[m].mean()
        dataset_rows.append(row)

    by_dataset = pd.DataFrame(dataset_rows).sort_values(["approach", "system", "dataset"])
    by_system = by_dataset.groupby(["approach", "system"], as_index=False).agg(
        num_datasets=("dataset", "count"),
        mape_mean=("mape_mean", "mean"),
        mae_mean=("mae_mean", "mean"),
        rmse_mean=("rmse_mean", "mean"),
    )

    return by_dataset, by_system


def stats(raw, group_cols):
    # wilcoxon signed-rank test, paired by repeat so the same seeds are compared
    pair_cols = group_cols + ["repeat"]
    paired = raw.groupby(["approach"] + pair_cols, as_index=False)[METRICS].mean()

    base = paired[paired["approach"] == "baseline"].drop(columns="approach")
    improved = paired[paired["approach"] == "improved"].drop(columns="approach")
    merged = base.merge(improved, on=pair_cols, suffixes=("_baseline", "_improved"))

    rows = []
    for keys, group in merged.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["paired_runs"] = len(group)

        for m in METRICS:
            b_vals = group[f"{m}_baseline"]
            i_vals = group[f"{m}_improved"]
            diffs = b_vals - i_vals

            result = wilcoxon(b_vals, i_vals)
            p = result.pvalue

            row[f"{m}_median_baseline"] = b_vals.median()
            row[f"{m}_median_improved"] = i_vals.median()
            row[f"{m}_median_difference"] = diffs.median()
            row[f"{m}_p_value"] = p
            row[f"{m}_significant_0_05"] = p <= 0.05

            if diffs.median() > 0:
                row[f"{m}_better"] = "improved"
            elif diffs.median() < 0:
                row[f"{m}_better"] = "baseline"
            else:
                row[f"{m}_better"] = "tie"

        rows.append(row)

    return pd.DataFrame(rows)


def save_all(out_dir, raw, by_dataset, by_system):
    baseline_raw = raw[raw["approach"] == "baseline"]
    improved_raw = raw[raw["approach"] == "improved"]

    baseline_dataset = by_dataset[by_dataset["approach"] == "baseline"]
    improved_dataset = by_dataset[by_dataset["approach"] == "improved"]

    baseline_system = by_system[by_system["approach"] == "baseline"]
    improved_system = by_system[by_system["approach"] == "improved"]

    # raw results
    baseline_raw.to_csv(out_dir / "baseline_raw_results.csv", index=False)
    improved_raw.to_csv(out_dir / "improved_raw_results.csv", index=False)
    raw.to_csv(out_dir / "combined_raw_results.csv", index=False)

    # summaries
    baseline_dataset.to_csv(out_dir / "baseline_summary_by_dataset.csv", index=False)
    improved_dataset.to_csv(out_dir / "improved_summary_by_dataset.csv", index=False)
    by_dataset.to_csv(out_dir / "combined_summary_by_dataset.csv", index=False)

    baseline_system.to_csv(out_dir / "baseline_summary_by_system.csv", index=False)
    improved_system.to_csv(out_dir / "improved_summary_by_system.csv", index=False)
    by_system.to_csv(out_dir / "combined_summary_by_system.csv", index=False)

    keep_dataset = ["system", "dataset"] + [f"{m}_mean" for m in METRICS]
    base_data = baseline_dataset[keep_dataset]
    imp_data = improved_dataset[keep_dataset]
    dataset_comp = base_data.merge(imp_data, on=["system", "dataset"], suffixes=("_baseline", "_improved"))

    for m in METRICS:
        base_col = f"{m}_mean_baseline"
        imp_col = f"{m}_mean_improved"
        dataset_comp[f"{m}_difference"] = dataset_comp[base_col] - dataset_comp[imp_col]
        dataset_comp[f"{m}_improvement_percent"] = dataset_comp[f"{m}_difference"] / dataset_comp[base_col] * 100

    keep_system = ["system"] + [f"{m}_mean" for m in METRICS]
    base_sys = baseline_system[keep_system]
    imp_sys = improved_system[keep_system]
    system_comp = base_sys.merge(imp_sys, on=["system"], suffixes=("_baseline", "_improved"))

    for m in METRICS:
        base_col = f"{m}_mean_baseline"
        imp_col = f"{m}_mean_improved"
        system_comp[f"{m}_difference"] = system_comp[base_col] - system_comp[imp_col]
        system_comp[f"{m}_improvement_percent"] = system_comp[f"{m}_difference"] / system_comp[base_col] * 100

    # comparisons + stats
    dataset_comp.to_csv(out_dir / "comparison_by_dataset.csv", index=False)
    system_comp.to_csv(out_dir / "comparison_by_system.csv", index=False)
    stats(raw, ["system", "dataset"]).to_csv(out_dir / "stats_by_dataset.csv", index=False)
    stats(raw, ["system"]).to_csv(out_dir / "stats_by_system.csv", index=False)




def main():
    root = Path(DATASET_DIR)
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(exist_ok=True)

    rows = []
    for system in SYSTEMS:
        folder = root / system

        if not folder.exists():
            print(f"WARNING: could not find folder for {system}, skipping")
            continue

        csv_files = sorted(folder.glob("*.csv"))

        for csv_path in csv_files:
            print(f"running: {system} -> {csv_path.name}")
            new_rows = run_set(system, csv_path)
            rows.extend(new_rows)

    raw = pd.DataFrame(rows)

    #print("\nprogram finished")
    by_dataset, by_system = make_summaries(raw)

    save_all(out_dir, raw, by_dataset, by_system)

    print("\n========== RESULTS SUMMARY ==========")
    for approach, group in by_system.groupby("approach"):
        print(f"\n[{approach}]")
        for _, row in group.iterrows():
            print(f"  {row['system']}: MAPE={row['mape_mean']:.4f}  MAE={row['mae_mean']:.4f}  RMSE={row['rmse_mean']:.4f}")

    print(f"\nresults saved in: {out_dir}")


if __name__ == "__main__":
    main()
