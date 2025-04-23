#!/usr/bin/env python3
"""
train.py

End-to-end training of a “set-of-sequences” classifier:
1) Load df, convert or load precomputed 64-d set embeddings
2) Feed (N×64) into an MLP to predict binary label
3) Train/validate loop with logging, checkpointing
"""

import argparse
import logging
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ─── Your imported modules ──────────────────────────────────────────────────────
from models.set_of_sequences import SetOfSequences

# ─── Seed everything for reproducibility ────────────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn deterministic behavior (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─── Dataset Definition ─────────────────────────────────────────────────────────
class CartDataset(Dataset):
    """
    Wraps a pandas DataFrame with:
      - products_before: List[str]
      - order_after: int (0/1)
      - optionally cached embeddings on disk for speed
    """
    def __init__(
        self,
        df: pd.DataFrame,
        transformer_model_name: str,
        device: torch.device,
        hidden_dim: int,
        output_dim: int,
        cache_dir: str = "cache_embeddings",
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.device = device
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        # Set up your set-of-sequences encoder
        self.model = SetOfSequences(
            transformer_model_name=transformer_model_name,
            device=self.device,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        self.model.eval()  # only inference here

    def __len__(self):
        return len(self.df)

    def _embed(self, idx: int) -> np.ndarray:
        """Compute or load 64-d embedding for row idx."""
        cache_path = os.path.join(self.cache_dir, f"{idx}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

        products: List[str] = self.df.loc[idx, "products_before"]
        with torch.no_grad():
            rep = self.model(products)           # torch.Tensor [output_dim]
        rep = rep.cpu().numpy()
        np.save(cache_path, rep)
        return rep

    def __getitem__(self, idx: int):
        # X: numpy array [64,], y: int
        x = self._embed(idx)
        y = int(self.df.loc[idx, "order_after"])
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

# ─── MLP Classifier ────────────────────────────────────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ─── Training & Validation Loops ───────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_X.size(0)
    return total_loss / len(dataloader.dataset)

# ─── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    seed_everything(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load your CSV (or pickle) into pandas
    df = pd.read_csv(args.data_path)
    # Ensure list-of-str column is real Python lists
    import ast
    df["products_before"] = df["products_before"].apply(ast.literal_eval)

    # Build dataset and splits
    dataset = CartDataset(
        df,
        transformer_model_name=args.transformer_name,
        device=args.device,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        cache_dir=args.cache_dir,
    )
    val_size = int(len(dataset) * args.val_frac)
    train_ds, val_ds = random_split(dataset, [len(dataset)-val_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build classifier, criterion, optimizer
    classifier = MLPClassifier(
        input_dim=args.output_dim,
        hidden_dim=args.classifier_hidden_dim,
        num_classes=2,
    ).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            classifier, train_loader, criterion, optimizer, args.device
        )
        val_loss = evaluate(classifier, val_loader, criterion, args.device)
        logging.info(
            f"Epoch {epoch}/{args.epochs} — "
            f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
        )

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(classifier.state_dict(), ckpt_path)
            logging.info(f"Saved new best model to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train set-of-sequences classifier")
    parser.add_argument("--data-path",      type=str,   required=True)
    parser.add_argument("--output-dir",     type=str,   default="checkpoints")
    parser.add_argument("--cache-dir",      type=str,   default="cache_embeddings")
    parser.add_argument("--transformer-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--device",         type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--val-frac",       type=float, default=0.1,
                        help="Fraction of data to keep for validation")
    parser.add_argument("--num-workers",    type=int,   default=4)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--hidden-dim",     type=int,   default=128,
                        help="Hidden dim for DeepSets phi/rho")
    parser.add_argument("--output-dim",     type=int,   default=64,
                        help="Output dim for DeepSets (and MLP input dim)")
    parser.add_argument("--classifier-hidden-dim", type=int, default=32)
    args = parser.parse_args()

    main(args)
