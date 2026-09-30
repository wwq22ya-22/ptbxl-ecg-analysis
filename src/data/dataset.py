"""Lazy loading dataset for PTB-XL records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ptbxl import SUPERCLASSES, PTBXLDataError, load_ecg


class PTBXLSignalDataset:
    """Map-style, lazy dataset returning `[12, T]` signal and multi-hot labels.

    The object follows the ``__len__``/``__getitem__`` protocol used by
    ``torch.utils.data.DataLoader`` without importing PyTorch in the data layer.
    NumPy arrays are converted to tensors automatically by PyTorch's default
    collate function.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        dataset_root: str | Path,
        *,
        sampling_rate_hz: int = 100,
        superclasses: tuple[str, ...] = SUPERCLASSES,
    ) -> None:
        filename_column = "filename_lr" if sampling_rate_hz == 100 else "filename_hr"
        required = {"ecg_id", "patient_id", filename_column, *superclasses}
        missing = sorted(required.difference(metadata.columns))
        if missing:
            raise PTBXLDataError(
                "Dataset metadata is missing columns: "
                f"{', '.join(missing)}. Add superclass labels before constructing the dataset."
            )
        if sampling_rate_hz not in (100, 500):
            raise ValueError("sampling_rate_hz must be 100 or 500 for PTB-XL.")
        self.metadata = metadata.reset_index(drop=True)
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.sampling_rate_hz = sampling_rate_hz
        self.superclasses = tuple(superclasses)
        self.filename_column = filename_column

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        record = load_ecg(
            self.dataset_root,
            str(row[self.filename_column]),
            sampling_rate_hz=self.sampling_rate_hz,
        )
        return {
            "signal": np.ascontiguousarray(record.signal.T, dtype=np.float32),
            "label": row.loc[list(self.superclasses)].to_numpy(dtype=np.float32),
            "ecg_id": int(row["ecg_id"]),
            "patient_id": int(row["patient_id"]),
            "sampling_rate_hz": record.sampling_rate_hz,
            "lead_names": record.lead_names,
            "units": record.units,
        }
