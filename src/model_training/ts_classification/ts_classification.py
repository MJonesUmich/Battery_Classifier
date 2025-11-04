import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

FEATURE_COLS = [
    "Current(A)",
    "Voltage(V)",
    "Test_Time(s)",
    "Cycle_Count",
    "Delta_Time(s)",
    "Delta_Ah",
    "Ah_throughput",
    "EFC",
]


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        split: str,
        chemistries: List[str],
        scaler: StandardScaler = None,
        encoder: LabelEncoder = None,
    ):
        self.files: List[str] = []
        self.labels: List[str] = []
        self.chemistries = chemistries

        # collect files from base_dir/split/<chem>/*.csv
        for chem in chemistries:
            dir1 = os.path.join(base_dir, split, chem)
            dir2 = os.path.join(base_dir, chem)
            csvs = []
            if os.path.isdir(dir1):
                csvs = glob.glob(os.path.join(dir1, "*.csv"))
            elif os.path.isdir(dir2):
                csvs = glob.glob(os.path.join(dir2, "*.csv"))
            for f in csvs:
                try:
                    hdr = pd.read_csv(f, nrows=1)
                    hdr_cols = set(hdr.columns.astype(str))
                    if not set(FEATURE_COLS) <= hdr_cols:
                        print(f"Skipping (missing cols) {f}")
                        continue
                except Exception:
                    print(f"Skipping unreadable file {f}")
                    continue
                self.files.append(f)
                self.labels.append(chem)

        # encoder/scaler
        self.encoder = encoder or LabelEncoder()
        if encoder is None and self.labels:
            self.encoder.fit(self.labels)
        self.labels_enc = self.encoder.transform(self.labels) if self.labels else []

        self.scaler = scaler or StandardScaler()
        # fit scaler on train only (caller should supply scaler fitted on train)

    def fit_scaler_from_files(self, files: List[str]):
        all_data = []
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if not set(FEATURE_COLS) <= set(df.columns.astype(str)):
                continue
            data = df[FEATURE_COLS].dropna().values
            if data.size:
                all_data.append(data)
        if not all_data:
            raise RuntimeError("No data available to fit scaler")
        concatenated = np.vstack(all_data)
        self.scaler.fit(concatenated)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        f = self.files[idx]
        label = self.labels[idx]
        df = pd.read_csv(f)
        data = df[FEATURE_COLS].dropna().values  # shape (seq_len, features)
        # transform features
        data = self.scaler.transform(data)
        # return tensor shaped (features, seq_len)
        tensor = torch.tensor(data, dtype=torch.float32).T
        label_idx = int(self.encoder.transform([label])[0])
        return tensor, label_idx


def pad_collate(batch):
    # batch: list of (tensor (F, L), label)
    lengths = [item[0].shape[1] for item in batch]
    max_len = max(lengths)
    feat = batch[0][0].shape[0]
    batch_size = len(batch)
    out = torch.zeros((batch_size, feat, max_len), dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.long)
    for i, (t, lbl) in enumerate(batch):
        L = t.shape[1]
        out[i, :, :L] = t
        labels[i] = lbl
    return out, labels


class CNN1DModel(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )
    chemistries = ["LFP", "NMC", "LCO", "NCA"]

    # datasets
    train_ds = TimeSeriesDataset(args.base_dir, "train", chemistries)
    if len(train_ds) == 0:
        raise RuntimeError(
            "No training files found. Check base_dir and train folder layout."
        )
    # fit scaler using train files
    train_ds.fit_scaler_from_files(train_ds.files)
    val_ds = TimeSeriesDataset(
        args.base_dir,
        "val",
        chemistries,
        scaler=train_ds.scaler,
        encoder=train_ds.encoder,
    )
    test_ds = TimeSeriesDataset(
        args.base_dir,
        "test",
        chemistries,
        scaler=train_ds.scaler,
        encoder=train_ds.encoder,
    )

    print(
        f"Train files: {len(train_ds.files)}, Val files: {len(val_ds.files)}, Test files: {len(test_ds.files)}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=pad_collate)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, collate_fn=pad_collate
    )

    # sample to get feature size
    sample_x, _ = train_ds[0]
    num_features = sample_x.shape[0]
    num_classes = len(train_ds.encoder.classes_)

    model = CNN1DModel(num_features, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch}: Train Loss {avg_loss:.4f}, Val Acc {val_acc:.4f}")

        # save best
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler": train_ds.scaler,
                    "encoder_classes": train_ds.encoder.classes_,
                },
                args.checkpoint,
            )
            print(f"Saved best model to {args.checkpoint} (val acc {best_val:.4f})")

    # final test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {test_acc:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-dir", default=r"D:\\Capstone\\Battery_Classifier\\continuous_model_prep"
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--use-cuda", action="store_true")
    p.add_argument("--checkpoint", default="ts_classifier_best.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

# ⏱️  Total processing time: 00:34:17
