"""Step 2b: the shared loader. Turns rows of the split into tensors a model can read.

One item is a dictionary:
    "aia"   : float tensor [number of AIA channels, H, W], normalised
    "hmi"   : float tensor [number of HMI channels, H, W], normalised
    "label" : float tensor, 1.0 = flare >= M in the next 24 h, else 0.0
    "id"    : the sample name (to trace predictions back to splits.csv)

Use in training code
    from src.data.dataset import make_loader, load_config
    config = load_config()
    train_loader = make_loader(config, "train")        # 4 training folds, shuffled, flipped at random
    val_loader   = make_loader(config, "validation")   # the validation fold
    test_loader  = make_loader(config, "test")         # only at the very end

Self-check (prints sizes and one batch)
    python src/data/dataset.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split import find_data_root  # noqa: E402

PARTS = ("train", "validation", "test")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class SolarDataset(Dataset):
    def __init__(self, config: dict, part: str, augment: bool = False, data_root: str = None):
        assert part in PARTS, f"part must be one of {PARTS}"
        self.data_root = find_data_root(data_root or config["data_root"])
        self.aia_channels = [str(c) for c in config["aia_channels"]]
        self.hmi_channels = [str(c) for c in config["hmi_channels"]]
        self.image_size = int(config["image_size"])
        self.augment = augment
        with open(config["norm_file"]) as f:
            self.norm = json.load(f)

        splits = pd.read_csv(config["splits_file"], dtype={"ar": str})
        index = pd.read_csv(config["index_file"], dtype=str, keep_default_na=False)
        index["usable"] = index["usable"] == "True"
        table = splits.merge(index, on="id", validate="one_to_one")

        val_fold = int(config["val_fold"])
        if part == "test":
            keep = table.split == "test"
        elif part == "validation":
            keep = (table.split == "train") & (table.fold == val_fold)
        else:
            keep = (table.split == "train") & (table.fold != val_fold)
        self.n_dropped = int((keep & ~table.usable).sum())       # samples lost to missing / blank images
        self.table = table[keep & table.usable].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.table)

    @property
    def labels(self) -> np.ndarray:
        """All labels, without opening any image (for class weights and samplers)."""
        return self.table.y.to_numpy()

    def _read(self, relative_path: str, channel: str) -> torch.Tensor:
        image = Image.open(os.path.join(self.data_root, relative_path)).convert("L")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        return (pixels - self.norm[channel]["mean"]) / self.norm[channel]["std"]

    def __getitem__(self, i: int) -> dict:
        row = self.table.iloc[i]
        aia = torch.stack([self._read(row[c], c) for c in self.aia_channels])
        hmi = torch.stack([self._read(row[c], c) for c in self.hmi_channels])
        if self.augment and torch.rand(1).item() < 0.5:
            # vertical flip only, the same for both instruments so they stay aligned.
            # The dataset authors advise against horizontal flips and random crops.
            aia, hmi = aia.flip(-2), hmi.flip(-2)
        return {"aia": aia, "hmi": hmi, "label": torch.tensor(float(row.y)), "id": row.id}


def make_loader(config: dict, part: str, batch_size: int = None, data_root: str = None) -> DataLoader:
    """Training is shuffled and augmented; validation and test are neither."""
    is_train = part == "train"
    dataset = SolarDataset(config, part, augment=is_train, data_root=data_root)
    return DataLoader(
        dataset,
        batch_size=batch_size or int(config["batch_size"]),
        shuffle=is_train,
        num_workers=int(config["num_workers"]),
        drop_last=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-check of the loader.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    print(f"{'part':12s} {'samples':>8s} {'positives':>10s} {'% positive':>11s} {'dropped':>8s}")
    for part in PARTS:
        ds = SolarDataset(config, part, data_root=args.data_root)
        print(f"{part:12s} {len(ds):8d} {int(ds.labels.sum()):10d} {100 * ds.labels.mean():11.1f} {ds.n_dropped:8d}")

    batch = next(iter(make_loader(config, "train", batch_size=8, data_root=args.data_root)))
    print("\none training batch")
    for name in ("aia", "hmi"):
        t = batch[name]
        print(f"  {name:5s} shape {tuple(t.shape)}  mean {t.mean():+.2f}  std {t.std():.2f}  min {t.min():+.2f}  max {t.max():+.2f}")
    print(f"  label {batch['label'].tolist()}")
    print(f"  id    {batch['id'][0]} ...")
    assert torch.isfinite(batch["aia"]).all() and torch.isfinite(batch["hmi"]).all(), "NaN or inf in a batch"
    print("\nloader OK")
