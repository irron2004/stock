from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class MaskedAEDataset(Dataset):
    def __init__(self, sequences: np.ndarray, mask_ratio: float = 0.5, mask_len: int = 5):
        self.sequences = sequences.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.mask_len = mask_len

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def _make_mask(self, length: int) -> np.ndarray:
        mask = np.zeros(length, dtype=bool)
        n_masked = max(1, int(length * self.mask_ratio))
        remaining = n_masked
        while remaining > 0:
            span = min(self.mask_len, remaining)
            start = np.random.randint(0, max(1, length - span + 1))
            mask[start : start + span] = True
            remaining -= span
        return mask

    def __getitem__(self, index: int):
        seq = self.sequences[index].copy()
        _, length = seq.shape
        mask = self._make_mask(length)
        masked = seq.copy()
        masked[:, mask] = 0.0
        return (
            torch.from_numpy(masked),
            torch.from_numpy(seq),
            torch.from_numpy(mask.astype(np.float32)),
        )


class MAE1D(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(depth):
            layers.append(nn.Conv1d(ch, hidden, kernel_size=5, padding=2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))
            ch = hidden
        self.encoder = nn.Sequential(*layers)
        self.head = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, in_channels, kernel_size=3, padding=1),
        )
        self.embedding_dim = hidden

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(inputs)
        embedding = self.head(latent).squeeze(-1)
        reconstruction = self.decoder(latent)
        return embedding, reconstruction


def train_mae1d(
    sequences: np.ndarray,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    mask_ratio: float = 0.5,
    mask_len: int = 5,
    device: str | None = None,
) -> Tuple[MAE1D, np.ndarray]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MaskedAEDataset(sequences, mask_ratio=mask_ratio, mask_len=mask_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = MAE1D(in_channels=sequences.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="none")

    model.train()
    for _ in range(epochs):
        for masked, original, mask in loader:
            masked = masked.to(device)
            original = original.to(device)
            mask = mask.to(device)
            _, reconstruction = model(masked)
            mask_expanded = mask.unsqueeze(1).expand_as(reconstruction)
            loss = mse(reconstruction, original)
            loss = (loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-9)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        total = sequences.shape[0]
        if total == 0:
            return model, np.empty((0, model.embedding_dim), dtype=np.float32)

        inference_batch_size = max(1, min(1024, batch_size))
        embeddings = np.empty((total, model.embedding_dim), dtype=np.float32)
        offset = 0
        while offset < total:
            end = min(offset + inference_batch_size, total)
            batch_np = np.ascontiguousarray(sequences[offset:end], dtype=np.float32)
            batch = torch.from_numpy(batch_np).to(device)
            emb, _ = model(batch)
            embeddings[offset:end] = emb.cpu().numpy().astype(np.float32, copy=False)
            offset = end
    return model, embeddings


def build_sequence_tensor(
    df: pd.DataFrame,
    embed_cols: Sequence[str],
    lookback: int,
) -> Tuple[np.ndarray, List[Tuple[str, pd.Timestamp]]]:
    sequences: List[np.ndarray] = []
    keys: List[Tuple[str, pd.Timestamp]] = []
    grouped = df.groupby(level=0)
    for asset, sub in grouped:
        sub = sub.sort_index(level=1)
        values = sub[list(embed_cols)].to_numpy()
        if len(values) < lookback:
            continue
        for idx in range(lookback - 1, len(values)):
            window = values[idx - lookback + 1 : idx + 1].T
            sequences.append(window)
            keys.append((asset, sub.index.get_level_values(1)[idx]))
    if not sequences:
        return np.empty((0, len(embed_cols), lookback)), []
    return np.stack(sequences, axis=0), keys


def aggregate_embeddings_over_time(embeddings: np.ndarray, suffix: str = "emb") -> pd.DataFrame:
    frame = pd.DataFrame(embeddings)
    frame.columns = [f"{suffix}_{col}" for col in range(frame.shape[1])]
    return frame
