"""Deployment samples must keep signals, labels and ECG identifiers aligned."""
import numpy as np
import pandas as pd

from scripts.prepare_stm32 import select
from src.data.ptbxl import SUPERCLASSES


def test_selection_uses_positions_with_noncontiguous_split_indices():
    frame = pd.DataFrame(
        {"ecg_id": [101, 203, 405, 809], **{
            name: [0, 1, 0, 1] for name in SUPERCLASSES
        }}, index=[10, 30, 50, 70],
    )
    signals = np.arange(4 * 12 * 1000, dtype=np.float32).reshape(4, 12, 1000)
    x, y, ids, rows = select(signals, frame, 3, 42)
    np.testing.assert_array_equal(x, signals[rows])
    np.testing.assert_array_equal(ids, frame.iloc[rows].ecg_id)
    np.testing.assert_array_equal(y, frame.iloc[rows][list(SUPERCLASSES)])
