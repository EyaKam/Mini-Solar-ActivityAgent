"""Build the one split file that every model uses.

What it does
    1. Reads training/meta_data.csv and test/meta_data.csv.
    2. Adds the binary label y (flare >= threshold in the next 24 h).
    3. Cuts the TRAINING samples into validation folds so that an active region
       is never in two folds, with positives spread as evenly as possible.
    4. Leaves the TEST samples untouched (fold = -1). The dataset authors already
       put different active regions in test.
    5. Checks for leaks, prints the counts, writes results/splits.csv.

Run
    python src/data/split.py                      # uses config.yaml
    python src/data/split.py --data-root /path    # override the data folder

Output columns
    id, ar, split (train/test), fold (0..n-1, or -1 for test), peak_flux, cls, y, path
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

CLASS_ORDER = ["quiet", "B", "C", "M", "X"]


def flux_to_class(flux: float) -> str:
    """GOES letter of a peak flux in W/m^2. Below 1e-7 the dataset means 'nothing happened'."""
    if flux < 1e-7:
        return "quiet"
    if flux < 1e-6:
        return "B"
    if flux < 1e-5:
        return "C"
    if flux < 1e-4:
        return "M"
    return "X"


def find_data_root(configured: str) -> str:
    """Return the folder containing training/meta_data.csv, searching common places if needed."""
    if configured and os.path.exists(os.path.join(configured, "training", "meta_data.csv")):
        return configured
    for base in [configured, "/kaggle/input", "data", "."]:
        if not base or not os.path.isdir(base):
            continue
        hits = glob.glob(os.path.join(base, "**", "training", "meta_data.csv"), recursive=True)
        if hits:
            return os.path.dirname(os.path.dirname(sorted(hits, key=len)[0]))
    sys.exit(f"Dataset not found. Set data_root in config.yaml (tried: {configured})")


def load_meta(data_root: str, folder: str, split_name: str, threshold: float) -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(data_root, folder, "meta_data.csv"), parse_dates=["start", "end"])
    meta["split"] = split_name
    # id looks like 11389_2012_01_01_19_06_00_0 : region number, then the sample name
    meta["ar"] = meta["id"].str.split("_").str[0]
    meta["cls"] = meta["peak_flux"].map(flux_to_class)
    meta["y"] = (meta["peak_flux"] >= threshold).astype(int)
    meta["path"] = [os.path.join(folder, ar, sample_id.split("_", 1)[1]) for ar, sample_id in zip(meta["ar"], meta["id"])]
    return meta


def fold_unevenness(train: pd.DataFrame, folds: np.ndarray) -> float:
    """0 = perfectly even folds. Looks at samples, positives and flaring regions per fold."""
    t = train.assign(fold=folds)
    counts = [
        t.groupby("fold").size(),
        t.groupby("fold").y.sum(),
        t[t.y == 1].groupby("fold").ar.nunique().reindex(sorted(t.fold.unique()), fill_value=0),
    ]
    return float(sum((c.max() - c.min()) / c.mean() for c in counts))


def assign_folds(train: pd.DataFrame, n_folds: int, seed: int, n_tries: int = 30) -> np.ndarray:
    """One fold number per training sample; all samples of a region share the same fold.

    scikit-learn versions do not all balance the folds equally well, so we try several
    shuffles (seed, seed+1, ...) plus the unshuffled version and keep the most even one.
    """
    candidates = [StratifiedGroupKFold(n_splits=n_folds)]
    candidates += [StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + k) for k in range(n_tries)]
    best_folds, best_score = None, float("inf")
    for splitter in candidates:
        folds = np.full(len(train), -1)
        for fold, (_, val_index) in enumerate(splitter.split(train, train["y"], groups=train["ar"])):
            folds[val_index] = fold
        score = fold_unevenness(train, folds)
        if score < best_score:
            best_folds, best_score = folds, score
    return best_folds


def check_no_leak(splits: pd.DataFrame) -> None:
    """Stop with an error if any active region appears in more than one place."""
    train, test = splits[splits.split == "train"], splits[splits.split == "test"]
    shared = set(train.ar) & set(test.ar)
    assert not shared, f"Regions in both train and test: {sorted(shared)[:5]}"
    folds_per_region = train.groupby("ar").fold.nunique()
    assert (folds_per_region == 1).all(), "A region is in two validation folds"
    assert (train.fold >= 0).all() and (test.fold == -1).all(), "Fold numbers are wrong"
    assert splits.id.is_unique, "Duplicate sample ids"


def summarise(splits: pd.DataFrame) -> pd.DataFrame:
    """Counts per fold: this is the table to paste into the report."""
    rows = []
    for name, part in [(f"fold {f}", splits[splits.fold == f]) for f in sorted(splits.fold[splits.fold >= 0].unique())] + [
        ("train (all folds)", splits[splits.split == "train"]),
        ("test", splits[splits.split == "test"]),
    ]:
        counts = part.cls.value_counts()
        rows.append({
            "part": name,
            "samples": len(part),
            "regions": part.ar.nunique(),
            "positives": int(part.y.sum()),
            "% positive": round(100 * part.y.mean(), 1),
            "flaring regions": part[part.y == 1].ar.nunique(),
            **{c: int(counts.get(c, 0)) for c in CLASS_ORDER},
        })
    return pd.DataFrame(rows).set_index("part")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen train/validation/test split.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default=None, help="overrides data_root from the config")
    parser.add_argument("--out", default=None, help="overrides splits_file from the config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    threshold = float(config["flare_threshold"])
    data_root = find_data_root(args.data_root or config["data_root"])
    out_path = args.out or config["splits_file"]

    train = load_meta(data_root, "training", "train", threshold)
    test = load_meta(data_root, "test", "test", threshold)
    train["fold"] = assign_folds(train, int(config["n_folds"]), int(config["seed"]))
    test["fold"] = -1

    splits = pd.concat([train, test], ignore_index=True)[["id", "ar", "split", "fold", "peak_flux", "cls", "y", "path"]]
    check_no_leak(splits)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    splits.to_csv(out_path, index=False)

    print(f"data root : {data_root}")
    print(f"label     : y = 1 if peak_flux >= {threshold:g}")
    print(f"seed      : {config['seed']}   folds: {config['n_folds']}   leak checks: passed\n")
    print(summarise(splits).to_string())
    print(f"\nwrote {len(splits)} rows to {out_path}")


if __name__ == "__main__":
    main()
