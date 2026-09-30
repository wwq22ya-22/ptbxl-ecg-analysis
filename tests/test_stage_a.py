from __future__ import annotations

import sys
import types

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from src.data.ptbxl import (
    ECGRecord,
    LEAD_NAMES,
    PTBXLDataError,
    create_fold_splits,
    encode_superclass_labels,
    find_ptbxl_root,
    load_ecg,
    parse_scp_codes,
)
from src.data.dataset import PTBXLSignalDataset
from src.visualization.ecg_plot import plot_12_leads, plot_single_lead


def make_dataset_root(tmp_path):
    root = tmp_path / "ptb-xl"
    (root / "records100").mkdir(parents=True)
    (root / "ptbxl_database.csv").write_text("", encoding="utf-8")
    (root / "scp_statements.csv").write_text("", encoding="utf-8")
    return root


def test_find_root_and_clear_missing_dataset_error(tmp_path):
    root = make_dataset_root(tmp_path)
    assert find_ptbxl_root(tmp_path) == root.resolve()

    with pytest.raises(PTBXLDataError, match="Official source"):
        find_ptbxl_root(tmp_path / "missing")


def test_scp_codes_are_safely_parsed_and_aggregated():
    statements = pd.DataFrame(
        {
            "diagnostic": [1, 1, 0],
            "diagnostic_class": ["MI", "STTC", "MI"],
        },
        index=["IMI", "NST_", "RHYTHM"])
    assert parse_scp_codes("{'IMI': 100, 'RHYTHM': 100}") == {"IMI": 100.0, "RHYTHM": 100.0}
    labels = encode_superclass_labels(
        {"IMI": 100, "NST_": 0, "RHYTHM": 100, "UNKNOWN": 100},
        statements,
    )
    np.testing.assert_array_equal(labels, [0, 1, 1, 0, 0])


def test_official_fold_split_is_patient_disjoint():
    rows = pd.DataFrame(
        {
            "patient_id": [10, 11, 12, 13],
            "strat_fold": [1, 8, 9, 10],
        }
    )
    split = create_fold_splits(rows, train_folds=(1, 8))
    assert list(split["train"]["patient_id"]) == [10, 11]
    assert list(split["validation"]["patient_id"]) == [12]
    assert list(split["test"]["patient_id"]) == [13]

    leaked = pd.DataFrame({"patient_id": [7, 7, 8], "strat_fold": [1, 9, 10]})
    with pytest.raises(PTBXLDataError, match="Patient leakage"):
        create_fold_splits(leaked, train_folds=(1,))


def test_wfdb_loader_reorders_leads_and_returns_physical_metadata(tmp_path, monkeypatch):
    root = make_dataset_root(tmp_path)
    record_name = "records100/00000/00001_lr"
    header = root / f"{record_name}.hea"
    header.parent.mkdir(parents=True)
    header.write_text("test header", encoding="utf-8")
    shuffled = list(reversed(LEAD_NAMES))
    signal = np.tile(np.arange(12, dtype=np.float32), (50, 1))
    fake_wfdb = types.SimpleNamespace(
        rdsamp=lambda _: (
            signal,
            {"fs": 100, "sig_name": shuffled, "sig_units": ["mV"] * 12},
        )
    )
    monkeypatch.setitem(sys.modules, "wfdb", fake_wfdb)

    record = load_ecg(root, record_name, sampling_rate_hz=100)
    assert record.signal.shape == (50, 12)
    assert record.lead_names == LEAD_NAMES
    assert record.units == ("mV",) * 12
    np.testing.assert_array_equal(record.signal[0], np.arange(11, -1, -1))


def test_dataset_returns_channel_first_signal_and_multilabel_vector(tmp_path, monkeypatch):
    import src.data.dataset as dataset_module

    signal = np.zeros((20, 12), dtype=np.float32)
    monkeypatch.setattr(
        dataset_module,
        "load_ecg",
        lambda *args, **kwargs: ECGRecord(
            signal=signal,
            sampling_rate_hz=100,
            lead_names=LEAD_NAMES,
            units=("mV",) * 12,
            record_name="record",
        ),
    )
    metadata = pd.DataFrame(
        [{"ecg_id": 3, "patient_id": 8, "filename_lr": "records100/x", "NORM": 1, "MI": 0,
          "STTC": 1, "CD": 0, "HYP": 0}]
    )

    sample = PTBXLSignalDataset(metadata, tmp_path)[0]
    assert sample["signal"].shape == (12, 20)
    np.testing.assert_array_equal(sample["label"], [1, 0, 1, 0, 0])
    assert sample["ecg_id"] == 3


def test_ecg_plots_use_seconds_and_twelve_leads():
    signal = np.zeros((1000, 12), dtype=np.float32)
    figure, axis = plot_single_lead(signal, 100, lead="II")
    assert axis.get_xlabel() == "Time (s)"
    assert axis.lines[0].get_xdata()[-1] == pytest.approx(9.99)
    figure.clf()

    figure = plot_12_leads(signal, 100)
    assert len(figure.axes) == 12
    assert sum(axis.get_xlabel() == "Time (s)" for axis in figure.axes) == 2
    figure.clf()
