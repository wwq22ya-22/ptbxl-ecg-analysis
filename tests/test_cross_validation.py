"""Leakage checks for configurable folds and record-local cache subsets."""
import numpy as np
import pandas as pd
import pytest
from src.data.ptbxl import create_fold_splits, PTBXLDataError, ECGRecord, LEAD_NAMES
from src.training.data import ensure_cache_subset, ensure_preprocessed_cache, load_labeled_splits
from src.signal_processing.filters import FilterConfig


def test_two_fold_holdout_is_disjoint_and_excludes_fold_10():
    frame = pd.DataFrame({'patient_id': range(10), 'strat_fold': range(1, 11)})
    for heldout in ((1,2), (3,4), (5,6), (7,8)):
        training = [fold for fold in range(1,9) if fold not in heldout]
        splits = create_fold_splits(frame, train_folds=training, test_folds=heldout)
        assert set(splits['test'].strat_fold) == set(heldout)
        assert set(splits['train'].strat_fold) == set(training)
        assert set(splits['validation'].strat_fold) == {9}
        assert all(10 not in set(part.strat_fold) for part in splits.values())
    with pytest.raises(ValueError, match='disjoint'):
        create_fold_splits(frame, test_folds=(1,2))
    with pytest.raises(ValueError, match='empty'):
        create_fold_splits(frame, test_folds=())
    frame.loc[2, 'patient_id'] = frame.loc[0, 'patient_id']
    with pytest.raises(PTBXLDataError, match='Patient leakage'):
        create_fold_splits(frame, train_folds=(3,4,5,6,7,8), test_folds=(1,2))


def test_loader_honors_configured_training_folds(monkeypatch):
    import src.training.data as module
    frame = pd.DataFrame({'patient_id': range(10), 'ecg_id': range(1,11), 'strat_fold': range(1,11), 'scp_codes': ["{'A': 100}"]*10})
    statements = pd.DataFrame({'diagnostic': [1], 'diagnostic_class': ['NORM']}, index=['A'])
    monkeypatch.setattr(module, 'load_ptbxl_metadata', lambda *args, **kwargs: frame)
    monkeypatch.setattr(module, 'load_scp_statements', lambda *args, **kwargs: statements)
    splits = load_labeled_splits('unused',100,train_folds=(3,4,5,6,7,8),test_folds=(1,2))
    assert splits['train'].ecg_id.tolist() == [3,4,5,6,7,8]
    assert splits['test'].ecg_id.tolist() == [1,2]


def test_subset_preserves_order_and_rejects_global_cache_reuse(tmp_path, monkeypatch):
    import src.training.data as module
    frame = pd.DataFrame({'ecg_id': [1,2,3], 'filename_lr': ['a','b','c']})
    waves = np.random.default_rng(2).normal(size=(3,1000,12)).astype(np.float32)
    monkeypatch.setattr(module,'load_ecg', lambda root,name,**kwargs: ECGRecord(waves['abc'.index(name)],100,LEAD_NAMES,('mV',)*12,name))
    cfg = FilterConfig(normalization='none')
    source = ensure_preprocessed_cache(frame,tmp_path,tmp_path/'source',split_name='train_full',sampling_rate_hz=100,config=cfg)
    selected = frame.iloc[[2,0]].copy()
    path = ensure_cache_subset(source,frame,selected,tmp_path/'subset',split_name='train_full',sampling_rate_hz=100,config=cfg)
    np.testing.assert_array_equal(np.load(path), np.load(source)[[2,0]])
    with pytest.raises(ValueError, match='Source cache preprocessing'):
        ensure_cache_subset(source,frame,selected,tmp_path/'other',split_name='train_full',sampling_rate_hz=100,config=FilterConfig(normalization='zscore'))
    with pytest.raises(ValueError, match='Subset only record-local'):
        ensure_cache_subset(source,frame,selected,tmp_path/'other',split_name='train_full',sampling_rate_hz=100,config=FilterConfig(normalization='train_global'))
