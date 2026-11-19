"""Step 5: the HMI CNN (Person 1). Predicts a flare >= M in the next 24 h from the magnetogram only.

What it does
    1. reads the shared loader (src/data/dataset.py); the model only uses batch["hmi"],
    2. trains a small CNN from scratch on every fold except `--val-fold`,
    3. after each epoch, scores the held-out fold with the SAME formulas as src/eval.py,
    4. keeps the best epoch (or the last one, see --select) and writes its predictions to
           results/preds/hmi_cnn_fold<k>_validation.csv      columns: id, prob
       (fold 0 is the official validation; folds 1 to 4 are the out-of-fold predictions for fusion).
    The test set stays closed unless --with-test is given (end of the project only).

Run
    python src/hmi_cnn.py                         # official run: validation = fold 0
    python src/eval.py --name hmi_cnn_fold0       # score it like every other model
    python src/hmi_cnn.py --val-fold 1            # out-of-fold run: leave fold 1 out
    python src/hmi_cnn.py --all-folds             # the five runs, one after the other

Class imbalance: about 6% of samples are positive. The loss gives each positive sample the weight
(number of negatives / number of positives) so that missing a flare costs as much, in total, as
a false alarm. The decision threshold is NOT tuned here: src/eval.py picks it on validation.
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "data"))
from dataset import SolarDataset, load_config, make_loader  # noqa: E402
from eval import best_threshold, metrics  # noqa: E402


# ------------------------------------------------------------------- model ----
def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class HmiCNN(nn.Module):
    """4 convolution blocks (256 -> 16 pixels), then average AND max pooling.

    Max pooling is kept next to average pooling because a flare-prone region is a small area of
    very strong field: the strongest response matters, not only the average one.
    """

    def __init__(self, in_channels: int = 1, width: int = 16, dropout: float = 0.3):
        super().__init__()
        c = [in_channels, width, 2 * width, 4 * width, 8 * width]
        self.features = nn.Sequential(*[conv_block(c[i], c[i + 1]) for i in range(4)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = torch.cat([z.mean(dim=(2, 3)), z.amax(dim=(2, 3))], dim=1)
        return self.head(z).squeeze(1)          # one logit per sample (sigmoid gives the probability)


# ---------------------------------------------------------------- training ----
@torch.no_grad()
def predict(model: nn.Module, loader, device: str) -> pd.DataFrame:
    model.eval()
    ids, probs, labels = [], [], []
    for batch in loader:
        logits = model(batch["hmi"].to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["label"].numpy())
        ids.extend(batch["id"])
    return pd.DataFrame({"id": ids, "prob": np.concatenate(probs), "y": np.concatenate(labels)})


def train_one_epoch(model, loader, loss_fn, optimizer, device: str) -> float:
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        x, y = batch["hmi"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


def write_predictions(table: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table[["id", "prob"]].to_csv(path, index=False)
    print(f"wrote {path}  ({len(table)} rows)", flush=True)


def run_fold(config: dict, val_fold: int, args, device: str) -> None:
    config = {**config, "val_fold": val_fold}            # the loader reads the validation fold from the config
    torch.manual_seed(int(config["seed"]) + val_fold)
    np.random.seed(int(config["seed"]) + val_fold)

    train_loader = make_loader(config, "train", data_root=args.data_root)
    val_loader = make_loader(config, "validation", data_root=args.data_root)
    labels = train_loader.dataset.labels
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    print(f"\n=== fold {val_fold} left out | train {len(labels)} ({n_pos} positives) | "
          f"validation {len(val_loader.dataset)} | device {device} ===", flush=True)

    model = HmiCNN(in_channels=len(config["hmi_channels"]), width=args.width).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = {"TSS": -np.inf, "epoch": 0, "state": None}
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        scheduler.step()
        val = predict(model, val_loader, device)
        threshold = best_threshold(val.y.to_numpy(), val.prob.to_numpy())
        m = metrics(val.y.to_numpy(), val.prob.to_numpy(), threshold)
        print(f"epoch {epoch:2d}/{args.epochs} | loss {loss:.4f} | val TSS {m['TSS']:.3f}  AUC {m['AUC']:.3f} "
              f"| {time.time() - started:.0f}s", flush=True)
        if args.select == "last" or m["TSS"] > best["TSS"]:
            best = {"TSS": m["TSS"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}

    model.load_state_dict(best["state"])
    print(f"kept epoch {best['epoch']} (validation TSS {best['TSS']:.3f}, selection: {args.select})")
    os.makedirs(args.models_dir, exist_ok=True)
    torch.save(best["state"], os.path.join(args.models_dir, f"hmi_cnn_fold{val_fold}.pt"))

    preds_dir = config.get("preds_dir", "results/preds")
    write_predictions(predict(model, val_loader, device), os.path.join(preds_dir, f"hmi_cnn_fold{val_fold}_validation.csv"))
    if args.with_test:                                    # end of the project only, by agreement
        test_loader = make_loader(config, "test", data_root=args.data_root)
        write_predictions(predict(model, test_loader, device), os.path.join(preds_dir, f"hmi_cnn_fold{val_fold}_test.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HMI CNN and write its predictions.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--val-fold", type=int, default=None, help="fold left out (default: val_fold of the config)")
    parser.add_argument("--all-folds", action="store_true", help="run every fold 0..n_folds-1, one after the other")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=16, help="channels of the first block (doubles at each block)")
    parser.add_argument("--select", choices=["best", "last"], default="best",
                        help="best: epoch with the highest validation TSS; last: always the last epoch (no peeking)")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--with-test", action="store_true", help="also write test predictions (end of project only)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.all_folds:
        folds = list(range(int(config["n_folds"])))
    else:
        folds = [args.val_fold if args.val_fold is not None else int(config["val_fold"])]
    for fold in folds:
        run_fold(config, fold, args, device)
    print("\nnext: python src/eval.py --name hmi_cnn_fold0")


if __name__ == "__main__":
    main()
