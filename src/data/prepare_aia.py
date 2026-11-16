"""Step 2a: find the images each sample will use, and measure how to normalise them.

For every sample in results/splits.csv and every channel in the config
(aia_channels and hmi_channels) this script:
    1. looks for the image of the LAST time step (about 10 min before the forecast window),
    2. rejects it if the file is missing or blank,
    3. records its path in results/index.csv,
    4. marks the sample `usable` if all channels were found.
It then computes each channel's mean and standard deviation on the TRAINING folds only
(never validation or test) and writes results/norm_stats.json.

It also indexes the magnetogram so the shared loader {hmi, aia, label} works; the HMI
normalisation itself is fixed (grey 128 = zero field) and belongs to Person 1.

Run (takes a few minutes: it opens about 28,000 images)
    python src/data/prepare_aia.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split import find_data_root  # noqa: E402


def read_end_times(data_root: str) -> pd.Series:
    """id -> end of the input window (the forecast window starts there)."""
    parts = [pd.read_csv(os.path.join(data_root, folder, "meta_data.csv"), parse_dates=["end"]) for folder in ["training", "test"]]
    return pd.concat(parts).set_index("id")["end"]


def last_step_files(sample_dir: str, end: pd.Timestamp, max_minutes: float) -> dict:
    """{channel: file name} for images taken within `max_minutes` before `end`.

    File names look like 2012-01-01T185600__171.jpg (timestamp, two underscores, channel).
    If a channel has several candidates, the one closest to `end` wins.
    """
    best = {}
    if not os.path.isdir(sample_dir):
        return best
    for name in os.listdir(sample_dir):
        if not name.endswith(".jpg") or "__" not in name:
            continue
        stamp, channel = name[:-4].split("__")
        minutes_before = (end - pd.to_datetime(stamp, format="%Y-%m-%dT%H%M%S")).total_seconds() / 60
        if -5 <= minutes_before <= max_minutes and (channel not in best or minutes_before < best[channel][0]):
            best[channel] = (minutes_before, name)
    return {channel: name for channel, (_, name) in best.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Index usable images and compute normalisation statistics.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_root = find_data_root(args.data_root or config["data_root"])
    channels = [str(c) for c in config["aia_channels"]] + [str(c) for c in config["hmi_channels"]]
    aia_channels = [str(c) for c in config["aia_channels"]]

    splits = pd.read_csv(config["splits_file"], dtype={"ar": str})
    end_times = read_end_times(data_root)
    is_train_fold = (splits.split == "train") & (splits.fold != int(config["val_fold"]))

    # running sums for mean / std, on a 0..1 pixel scale, training folds only
    stats = {c: {"n": 0, "sum": 0.0, "sum_sq": 0.0} for c in aia_channels}
    problems = {c: {"missing": 0, "blank": 0} for c in channels}
    rows = []

    for i, row in enumerate(splits.itertuples()):
        if i % 1000 == 0:
            print(f"  {i}/{len(splits)} samples checked", flush=True)
        sample_dir = os.path.join(data_root, row.path)
        files = last_step_files(sample_dir, end_times[row.id], float(config["last_step_max_minutes"]))
        entry = {"id": row.id}
        for channel in channels:
            entry[channel] = ""
            if channel not in files:
                problems[channel]["missing"] += 1
                continue
            pixels = np.asarray(Image.open(os.path.join(sample_dir, files[channel])).convert("L"), dtype=np.float32)
            if pixels.std() < float(config["blank_std"]):
                problems[channel]["blank"] += 1
                continue
            entry[channel] = os.path.join(row.path, files[channel])
            if channel in stats and is_train_fold.iloc[i]:
                scaled = pixels / 255.0
                stats[channel]["n"] += scaled.size
                stats[channel]["sum"] += float(scaled.sum())
                stats[channel]["sum_sq"] += float((scaled ** 2).sum())
        entry["usable"] = all(entry[c] != "" for c in channels)
        rows.append(entry)

    index = pd.DataFrame(rows)
    index.to_csv(config["index_file"], index=False)

    norm = {}
    for channel, s in stats.items():
        mean = s["sum"] / s["n"]
        norm[channel] = {"mean": round(mean, 5), "std": round(float(np.sqrt(s["sum_sq"] / s["n"] - mean ** 2)), 5)}
    for channel in config["hmi_channels"]:
        # grey 128 = zero field: fixed normalisation that keeps the sign of the field
        norm[str(channel)] = {"mean": round(128 / 255, 5), "std": round(128 / 255, 5)}
    with open(config["norm_file"], "w") as f:
        json.dump(norm, f, indent=2)

    # ---- report -------------------------------------------------------------
    merged = splits.merge(index[["id", "usable"]], on="id")
    merged["part"] = np.where(merged.split == "test", "test", np.where(merged.fold == int(config["val_fold"]), "validation", "train"))
    report = merged.groupby("part").agg(
        samples=("id", "size"), usable=("usable", "sum"), positives=("y", "sum"),
        usable_positives=("y", lambda y: int((y * merged.loc[y.index, "usable"]).sum())),
    ).reindex(["train", "validation", "test"])
    report["% usable"] = (100 * report.usable / report.samples).round(1)

    print(f"\ndata root : {data_root}")
    print(f"channels  : AIA {aia_channels} + HMI {config['hmi_channels']}, last time step only\n")
    print(report.to_string())
    print("\nrejected images per channel:")
    print(pd.DataFrame(problems).T.to_string())
    print("\nnormalisation (pixels scaled to 0..1, measured on training folds):")
    print(json.dumps(norm, indent=2))
    print(f"\nwrote {config['index_file']} and {config['norm_file']}")


if __name__ == "__main__":
    main()
