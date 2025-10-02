from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    horizon: int = 5
    windows: List[int] = field(default_factory=lambda: [21, 63, 126])
    ema_spans: List[int] = field(default_factory=lambda: [12, 26, 55])
    embed_cols: List[str] = field(
        default_factory=lambda: ["ret", "volume_z", "hl_range", "cl_pos_in_range"]
    )
    lookback: int = 60
    ssl_epochs: int = 10
    mask_ratio: float = 0.5
    mask_len: int = 5
    n_splits: int = 5
    embargo: int = 5
    cost_bps: float = 10.0
    top_k: float = 0.2
