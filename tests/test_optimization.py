import numpy as np
import pytest
import torch
from src.models import build_model
from src.training.data import fit_training_normalization
from src.signal_processing.filters import FilterConfig, preprocess_ecg


@pytest.mark.parametrize('kind', ['inception1d', 'cnn_bigru'])
def test_alternatives_backward_and_input_validation(kind):
    torch.set_num_threads(2)
    model = build_model(kind)
    logits = model(torch.randn(2, 12, 1000))
    assert logits.shape == (2, 5)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        model(torch.randn(2, 11, 1000))


def test_training_normalization_preserves_record_amplitude(tmp_path):
    values = np.random.default_rng(42).normal(size=(260, 12, 100)).astype(np.float32)
    values[0] *= 3
    path = tmp_path / 'train.npy'
    np.save(path, values)
    mean, std = fit_training_normalization(path)
    np.testing.assert_allclose(mean, values.astype(np.float64).mean(axis=(0, 2)), atol=1e-10)
    np.testing.assert_allclose(std, values.astype(np.float64).std(axis=(0, 2)), atol=1e-10)
    signal = np.random.default_rng(1).normal(size=(1000, 12)).astype(np.float32)
    raw = preprocess_ecg(signal, 100, config=FilterConfig(normalization='none'))
    transformed = preprocess_ecg(signal, 100, config=FilterConfig(normalization='train_global', training_mean=tuple(mean), training_std=tuple(std)))
    np.testing.assert_allclose(transformed, (raw - mean) / std, rtol=1e-5, atol=1e-6)

def test_global_cache_matches_single_record_preprocessing(tmp_path, monkeypatch):
    import pandas as pd
    from src.data.ptbxl import ECGRecord, LEAD_NAMES, SUPERCLASSES
    from src.training.data import ensure_preprocessed_cache
    import src.training.data as data_module
    signals = np.random.default_rng(11).normal(size=(2, 1000, 12)).astype(np.float32)
    records = {f'fake/{i}': ECGRecord(signals[i], 100, LEAD_NAMES, ('mV',)*12, f'fake/{i}') for i in range(2)}
    monkeypatch.setattr(data_module, 'load_ecg', lambda root, name, **kwargs: records[name])
    frame = pd.DataFrame({'ecg_id': [1, 2], 'filename_lr': ['fake/0', 'fake/1'], **{c: [0, 1] for c in SUPERCLASSES}})
    cfg = FilterConfig(normalization='train_global', training_mean=tuple(np.linspace(-0.1, 0.1, 12)), training_std=tuple(np.linspace(0.5, 2, 12)))
    cache = ensure_preprocessed_cache(frame, tmp_path, tmp_path/'cache', split_name='validation', sampling_rate_hz=100, config=cfg)
    batch = np.load(cache)
    for i in range(2):
        np.testing.assert_allclose(batch[i], preprocess_ecg(signals[i], 100, config=cfg).T, rtol=1e-5, atol=1e-6)
    assert ensure_preprocessed_cache(frame, tmp_path, tmp_path/'cache', split_name='validation', sampling_rate_hz=100, config=cfg) == cache
