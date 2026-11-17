"""Step 3: the one way every model is scored.

A model never scores itself. It only writes its predictions to a file:
    results/preds/<name>_validation.csv      columns: id, prob
    results/preds/<name>_test.csv            (only at the very end of the project)
where prob is the predicted probability of a flare >= M. This script does the rest:
    1. joins the predictions with results/splits.csv to get the true label and the region,
    2. picks the decision threshold that maximises TSS on VALIDATION,
    3. computes the metrics, with 95% confidence intervals obtained by resampling
       active regions (not samples: samples of one region are near-copies),
    4. applies the SAME threshold to the test predictions if they are given,
    5. prints a table and saves the rows in results/results.csv.

Run
    python src/eval.py --name aia_cnn                  # validation only
    python src/eval.py --name aia_cnn --with-test      # validation + test, end of project
    python src/eval.py --self-test                     # checks the formulas on known cases

Metrics
    TSS    = recall - false alarm rate. Main metric. 0 = no skill, 1 = perfect. Not affected by how rare flares are.
    HSS    = Heidke skill score: improvement over random guessing. Depends on the flare rate.
    AUC    = ranking quality, no threshold needed. 0.5 = random.
    F1, precision, recall: depend on the threshold and on the flare rate.
    Brier  = mean squared error of the probability. Only meaningful if probabilities are calibrated.
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------- formulas ----
def confusion(y: np.ndarray, pred: np.ndarray) -> tuple:
    """Counts (TP, FP, FN, TN) for true labels y and yes/no predictions pred."""
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum()), int((~y & ~pred).sum())


def tss(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / max(tp + fn, 1) - fp / max(fp + tn, 1)


def hss(tp: int, fp: int, fn: int, tn: int) -> float:
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return 2 * (tp * tn - fp * fn) / denominator if denominator else 0.0


def best_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """The probability cut-off with the highest TSS. Use on validation only."""
    false_alarm_rate, recall, thresholds = roc_curve(y, prob)
    best = int(np.argmax(recall - false_alarm_rate))
    return float(min(thresholds[best], 1.0))


def metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    y, prob = np.asarray(y), np.asarray(prob, dtype=float)
    tp, fp, fn, tn = confusion(y, prob >= threshold)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "TSS": tss(tp, fp, fn, tn),
        "HSS": hss(tp, fp, fn, tn),
        "AUC": roc_auc_score(y, prob) if 0 < y.sum() < len(y) else float("nan"),
        "F1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": fp / max(fp + tn, 1),
        "Brier": float(np.mean((prob - y) ** 2)),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def region_bootstrap(y, prob, regions, threshold: float, n_rounds: int = 1000, seed: int = 0) -> dict:
    """95% intervals for TSS and AUC. Each round redraws whole active regions with replacement."""
    y, prob, regions = np.asarray(y), np.asarray(prob, dtype=float), np.asarray(regions)
    rows_of = [np.flatnonzero(regions == r) for r in np.unique(regions)]
    rng = np.random.default_rng(seed)
    tss_values, auc_values = [], []
    for _ in range(n_rounds):
        rows = np.concatenate([rows_of[k] for k in rng.integers(len(rows_of), size=len(rows_of))])
        if 0 < y[rows].sum() < len(rows):                       # need both classes to score
            m = metrics(y[rows], prob[rows], threshold)
            tss_values.append(m["TSS"]); auc_values.append(m["AUC"])
    low, high = (lambda v: np.percentile(v, 2.5)), (lambda v: np.percentile(v, 97.5))
    return {"TSS_low": low(tss_values), "TSS_high": high(tss_values), "AUC_low": low(auc_values), "AUC_high": high(auc_values)}


# ------------------------------------------------------------------- files ----
def load_predictions(path: str, splits: pd.DataFrame) -> pd.DataFrame:
    preds = pd.read_csv(path)
    assert {"id", "prob"} <= set(preds.columns), f"{path} must have the columns id and prob"
    assert preds.id.is_unique, f"{path}: a sample id appears twice"
    assert preds.prob.between(0, 1).all(), f"{path}: prob must be between 0 and 1"
    table = preds.merge(splits[["id", "ar", "split", "fold", "y", "cls"]], on="id", how="left")
    assert table.y.notna().all(), f"{path}: some ids are not in the split file"
    return table


def score(table: pd.DataFrame, threshold: float) -> dict:
    result = metrics(table.y.to_numpy(), table.prob.to_numpy(), threshold)
    result.update(region_bootstrap(table.y, table.prob, table.ar, threshold))
    return {"samples": len(table), "positives": int(table.y.sum()), "regions": table.ar.nunique(), "threshold": threshold, **result}


def show(name: str, part: str, r: dict) -> None:
    print(f"\n{name} | {part}: {r['samples']} samples, {r['positives']} positives, {r['regions']} regions, threshold {r['threshold']:.3f}")
    print(f"  TSS  {r['TSS']:.3f}   95% interval [{r['TSS_low']:.3f}, {r['TSS_high']:.3f}]")
    print(f"  AUC  {r['AUC']:.3f}   95% interval [{r['AUC_low']:.3f}, {r['AUC_high']:.3f}]")
    print(f"  HSS  {r['HSS']:.3f}   F1 {r['F1']:.3f}   precision {r['precision']:.3f}   recall {r['recall']:.3f}   false alarm rate {r['false_alarm_rate']:.3f}")
    print(f"  Brier {r['Brier']:.4f}   (TP {r['TP']}, FP {r['FP']}, FN {r['FN']}, TN {r['TN']})")


def save(results_file: str, name: str, part: str, r: dict) -> None:
    """Add or replace the row (name, part) in the shared results table."""
    row = pd.DataFrame([{"model": name, "part": part, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}}])
    if os.path.exists(results_file):
        old = pd.read_csv(results_file)
        row = pd.concat([old[~((old.model == name) & (old.part == part))], row], ignore_index=True)
    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
    row.to_csv(results_file, index=False)


# --------------------------------------------------------------- self-test ----
def self_test() -> None:
    # hand-computed case: 10 flares, 90 non-flares; the model catches 8 flares and raises 9 false alarms
    tp, fp, fn, tn = 8, 9, 2, 81
    assert abs(tss(tp, fp, fn, tn) - (0.8 - 0.1)) < 1e-12
    assert abs(hss(tp, fp, fn, tn) - 2 * (8 * 81 - 9 * 2) / (10 * 83 + 17 * 90)) < 1e-12
    y = np.array([1] * 10 + [0] * 90)
    perfect = metrics(y, y.astype(float), 0.5)
    assert perfect["TSS"] == 1 and perfect["HSS"] == 1 and perfect["AUC"] == 1 and perfect["Brier"] == 0
    always_no = metrics(y, np.zeros(100), 0.5)
    assert always_no["TSS"] == 0 and always_no["HSS"] == 0, "predicting 'never' must have zero skill"
    always_yes = metrics(y, np.ones(100), 0.5)
    assert always_yes["TSS"] == 0 and always_yes["HSS"] == 0, "predicting 'always' must have zero skill"
    inverted = metrics(y, 1 - y.astype(float), 0.5)
    assert inverted["TSS"] == -1
    rng = np.random.default_rng(0)
    big_y = (rng.random(20000) < 0.06).astype(int)
    assert abs(metrics(big_y, rng.random(20000), 0.5)["TSS"]) < 0.05, "random guessing must be near zero"
    prob = np.clip(0.2 * big_y + rng.normal(0.3, 0.1, 20000), 0, 1)
    threshold = best_threshold(big_y, prob)
    assert metrics(big_y, prob, threshold)["TSS"] >= max(metrics(big_y, prob, t)["TSS"] for t in (0.2, 0.3, 0.4, 0.5)) - 1e-9
    interval = region_bootstrap(big_y[:2000], prob[:2000], np.arange(2000) // 10, threshold, n_rounds=200)
    assert interval["TSS_low"] <= metrics(big_y[:2000], prob[:2000], threshold)["TSS"] <= interval["TSS_high"]
    print("self-test passed: TSS, HSS, AUC, Brier, threshold choice and intervals behave as expected")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a model's prediction files.")
    parser.add_argument("--name", help="model name, as used in results/preds/<name>_validation.csv")
    parser.add_argument("--with-test", action="store_true", help="also score results/preds/<name>_test.csv with the validation threshold")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    assert args.name, "give --name, or use --self-test"

    with open(args.config) as f:
        config = yaml.safe_load(f)
    splits = pd.read_csv(config["splits_file"], dtype={"ar": str})
    preds_dir = config.get("preds_dir", "results/preds")
    results_file = config.get("results_file", "results/results.csv")

    validation = load_predictions(os.path.join(preds_dir, f"{args.name}_validation.csv"), splits)
    assert (validation.split == "train").all() and (validation.fold == int(config["val_fold"])).all(), \
        "the validation file contains samples that are not in the validation fold"
    threshold = best_threshold(validation.y.to_numpy(), validation.prob.to_numpy())
    result = score(validation, threshold)
    show(args.name, "validation", result); save(results_file, args.name, "validation", result)

    if args.with_test:
        test = load_predictions(os.path.join(preds_dir, f"{args.name}_test.csv"), splits)
        assert (test.split == "test").all(), "the test file contains samples that are not in the test set"
        result = score(test, threshold)                           # threshold comes from validation, never from test
        show(args.name, "test", result); save(results_file, args.name, "test", result)
    print(f"\nsaved to {results_file}")


if __name__ == "__main__":
    main()
