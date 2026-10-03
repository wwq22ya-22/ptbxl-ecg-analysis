"""Diagnostic-only cohorts keep mixed annotations and preserve official folds."""
import numpy as np
import pandas as pd
import src.training.data as module


def test_filter_drops_only_empty_diagnostic_targets(monkeypatch):
    metadata = pd.DataFrame({
        'ecg_id': [1, 2, 3, 4, 5, 6], 'patient_id': [11, 12, 13, 14, 15, 16],
        'strat_fold': [1, 1, 9, 9, 10, 10],
        'scp_codes': ["{'PACE':100}", "{'MI':100,'PACE':100}",
                      "{'AFIB':100}", "{'NORM':100}", "{'MI':0}", "{'ABQRS':100}"],
        'filename_lr': ['x'] * 6,
    })
    statements = pd.DataFrame({
        'diagnostic': [np.nan, 1, np.nan, 1, np.nan],
        'diagnostic_class': [np.nan, 'MI', np.nan, 'NORM', np.nan],
    }, index=['PACE', 'MI', 'AFIB', 'NORM', 'ABQRS'])
    monkeypatch.setattr(module, 'load_ptbxl_metadata', lambda *a, **k: metadata)
    monkeypatch.setattr(module, 'load_scp_statements', lambda *a, **k: statements)
    original = module.load_labeled_splits('.', 100, train_folds=[1])
    filtered = module.load_labeled_splits('.', 100, train_folds=[1], require_diagnostic_label=True)
    assert [len(original[k]) for k in ['train','validation','test']] == [2,2,2]
    assert [filtered[k].ecg_id.tolist() for k in ['train','validation','test']] == [[2],[4],[5]]
    assert filtered['train'].iloc[0].scp_codes == "{'MI':100,'PACE':100}"
    assert filtered['test'].iloc[0].MI == 1  # likelihood zero means unknown likelihood.
