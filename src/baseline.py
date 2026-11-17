"""The reference to beat: flare prediction from a few numbers per image, no deep learning.

For each sample it measures, on the same images the neural networks will see:
    AIA (each channel): average brightness, brightness of the brightest 1% of pixels, maximum
    HMI magnetogram   : average field strength, share of strong-field pixels, spread
and fits a logistic regression on the training folds. Three versions are produced so the
question "HMI, AIA or both?" gets a first answer before any neural network:
    baseline_aia, baseline_hmi, baseline_both

It writes prediction files; scoring is done by src/eval.py, like for every other model.

Run
    python src/baseline.py                 # validation predictions
    python src/baseline.py --with-test     # also test predictions (end of project only)
then
    python src/eval.py --name baseline_aia
    python src/eval.py --name baseline_hmi
    python src/eval.py --name baseline_both
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
from dataset import SolarDataset, load_config  # noqa: E402


def describe(item: dict) -> dict:
    """A handful of numbers summarising one sample's images."""
    features = {}
    for k, image in enumerate(item["aia"]):
        flat = image.flatten()
        features[f"aia{k}_mean"] = flat.mean().item()
        features[f"aia{k}_top1pct"] = flat.quantile(0.99).item()
        features[f"aia{k}_max"] = flat.max().item()
    for k, image in enumerate(item["hmi"]):
        strength = image.abs()                       # 0 = no field, 1 = saturated (250 gauss or more)
        features[f"hmi{k}_mean_strength"] = strength.mean().item()
        features[f"hmi{k}_strong_share"] = (strength > 0.8).float().mean().item()
        features[f"hmi{k}_spread"] = image.std().item()
    return features


def feature_table(config: dict, part: str, data_root: str, threads: int) -> pd.DataFrame:
    dataset = SolarDataset(config, part, augment=False, data_root=data_root)
    started = time.time()
    with ThreadPoolExecutor(max_workers=threads) as pool:        # the slow part is reading images from disk
        rows = list(pool.map(lambda i: describe(dataset[i]), range(len(dataset))))
    print(f"  {part}: {len(dataset)} samples described in {time.time() - started:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    table.insert(0, "id", dataset.table.id.values)
    table["y"] = dataset.labels
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature baseline (no deep learning).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--with-test", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    config = load_config(args.config)
    preds_dir = config.get("preds_dir", "results/preds")
    os.makedirs(preds_dir, exist_ok=True)

    parts = ["train", "validation"] + (["test"] if args.with_test else [])
    tables = {part: feature_table(config, part, args.data_root, args.threads) for part in parts}

    columns = [c for c in tables["train"].columns if c not in ("id", "y")]
    versions = {
        "baseline_aia": [c for c in columns if c.startswith("aia")],
        "baseline_hmi": [c for c in columns if c.startswith("hmi")],
        "baseline_both": columns,
    }
    for name, cols in versions.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
        model.fit(tables["train"][cols], tables["train"]["y"])
        for part in parts[1:]:
            out = pd.DataFrame({"id": tables[part]["id"], "prob": model.predict_proba(tables[part][cols])[:, 1]})
            path = os.path.join(preds_dir, f"{name}_{part}.csv")
            out.to_csv(path, index=False)
            print(f"wrote {path}  ({len(cols)} features)")


if __name__ == "__main__":
    main()
