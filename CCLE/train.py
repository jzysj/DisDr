# -*- coding: utf-8 -*-
"""Train the shared_consensus_common_complementary model."""

from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import datetime
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from load import load_dataset
from model import CellDrugRegressor


class RNAStandardizer:
    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_samples(
    dataset: dict,
    indices: Sequence[int],
    scaled_rna: np.ndarray,
):
    samples = []
    for local_index, global_index in enumerate(indices):
        global_index = int(global_index)
        graph = dataset["drug_graphs"][dataset["sample_drugs"][global_index]]
        sample = Data(
            x=graph.x,
            rna=torch.tensor(scaled_rna[local_index]).view(1, -1),
            y=torch.tensor([dataset["y"][global_index]], dtype=torch.float32),
        )
        for scale in range(4):
            setattr(
                sample,
                f"edge_index_s{scale}",
                getattr(graph, f"edge_index_s{scale}"),
            )
        samples.append(sample)
    return samples


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = prediction.astype(np.float64)
    target = target.astype(np.float64)
    error = prediction - target

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    r2 = float(
        1.0
        - np.sum(error ** 2)
        / (np.sum((target - target.mean()) ** 2) + 1e-12)
    )
    pcc = float(np.corrcoef(prediction, target)[0, 1])
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "PCC": pcc}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    targets = []

    for data in loader:
        data = data.to(device)
        prediction, _ = model(data)
        predictions.append(prediction.cpu().numpy())
        targets.append(data.y.view(-1).cpu().numpy())

    return regression_metrics(
        np.concatenate(predictions),
        np.concatenate(targets),
    )


def train_one_run(args, dataset, split, run_id, device):
    set_seed(args.seed + run_id)
    train_index, validation_index, test_index = split

    standardizer = RNAStandardizer().fit(dataset["X_rna"][train_index])
    train_set = build_samples(
        dataset,
        train_index,
        standardizer.transform(dataset["X_rna"][train_index]),
    )
    validation_set = build_samples(
        dataset,
        validation_index,
        standardizer.transform(dataset["X_rna"][validation_index]),
    )
    test_set = build_samples(
        dataset,
        test_index,
        standardizer.transform(dataset["X_rna"][test_index]),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = CellDrugRegressor(
        atom_dim=dataset["atom_dim"],
        rna_dim=dataset["rna_dim"],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.SmoothL1Loss(beta=args.smooth_l1_beta)

    best_rmse = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    print(f"\nrun {run_id}/{args.repeats}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        prediction_sum = 0.0
        alignment_sum = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            prediction, alignment = model(data)
            prediction_loss = criterion(prediction, data.y.view(-1))
            loss = prediction_loss + args.lambda_alignment * alignment

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_sum += loss.item()
            prediction_sum += prediction_loss.item()
            alignment_sum += alignment.item()

        validation_metrics = evaluate(model, validation_loader, device)
        validation_rmse = validation_metrics["RMSE"]
        n_batches = len(train_loader)

        if epoch == 1 or epoch % args.log_interval == 0 or validation_rmse < best_rmse:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss_sum / n_batches:.5f} "
                f"pred={prediction_sum / n_batches:.5f} "
                f"align={alignment_sum / n_batches:.5f} | "
                f"val RMSE={validation_metrics['RMSE']:.5f} "
                f"MAE={validation_metrics['MAE']:.5f} "
                f"R2={validation_metrics['R2']:.5f} "
                f"PCC={validation_metrics['PCC']:.5f}"
            )

        if validation_rmse < best_rmse - 1e-6:
            best_rmse = validation_rmse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    print(
        f"run {run_id} test | "
        f"RMSE={test_metrics['RMSE']:.6f} "
        f"MAE={test_metrics['MAE']:.6f} "
        f"R2={test_metrics['R2']:.6f} "
        f"PCC={test_metrics['PCC']:.6f} "
        f"best_epoch={best_epoch}"
    )

    return {
        "run": run_id,
        "best_epoch": best_epoch,
        **test_metrics,
    }


def summarize(rows):
    summary = {}
    for metric in ("RMSE", "MAE", "R2", "PCC"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def write_csv(path: str, rows) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_radii(text: str) -> Tuple[float, float, float, float]:
    values = tuple(float(value.strip()) for value in text.split(","))
    if len(values) != 4:
        raise ValueError("--radii requires four values.")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--response_csv",
        type=str,
        default="/home/jzy/model/MolDr/CCLE/data/CCLE/CCLE_response.csv",
    )
    parser.add_argument(
        "--rnaseq_csv",
        type=str,
        default="/home/jzy/model/MolDr/CCLE/data/CCLE/CCLE_RNAseq.csv",
    )
    parser.add_argument(
        "--coord_npy",
        type=str,
        default="/home/jzy/model/MolDr/CCLE/drug_3d_coords_H_all.npy",
    )
    parser.add_argument("--radii", default="2.0,2.5,3.0,3.5")

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--smooth_l1_beta", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--lambda_alignment", type=float, default=0.003)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--output_dir",
        default="runs_shared_consensus_common_complementary",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(
        response_csv=args.response_csv,
        rnaseq_csv=args.rnaseq_csv,
        coord_npy=args.coord_npy,
        radii=parse_radii(args.radii),
    )

    sample_index = np.arange(dataset["n_samples"])
    train_index, test_index = train_test_split(
        sample_index,
        test_size=0.1,
        random_state=np.random.randint(0, 1000),
    )
    train_index, validation_index = train_test_split(
        train_index,
        test_size=1 / 9,
        random_state=np.random.randint(0, 1000),
    )
    split = train_index, validation_index, test_index

    os.makedirs(args.output_dir, exist_ok=True)
    run_rows = [
        train_one_run(args, dataset, split, run_id, device)
        for run_id in range(1, args.repeats + 1)
    ]
    summary = summarize(run_rows)
    print("\nshared_consensus_common_complementary")
    for key, value in summary.items():
        print(f"{key}: {value:.8f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_csv(
        os.path.join(args.output_dir, f"run_results_{timestamp}.csv"),
        run_rows,
    )
    write_csv(
        os.path.join(args.output_dir, f"summary_{timestamp}.csv"),
        [summary],
    )


if __name__ == "__main__":
    main()
