"""Study-specific boundary checks for two populations and frozen selection."""
from pathlib import Path
import pandas as pd
import pytest
from scripts import run_unet_teacher as study

def guard():
    return study.UNetReadGuard(pd.DataFrame({'strat_fold': [1, 10],
        'filename_lr': ['records100/a', 'records100/b'], 'filename_hr': ['records500/a', 'records500/b']}))

def test_new_summary_exception_does_not_allow_foreign_results():
    g = guard(); g.check(study.OUT / 'summary.json')
    with pytest.raises(RuntimeError, match='prohibited'):
        g.check(study.ROOT / 'outputs/teacher_preprocessing/summary.json')

def test_new_output_directory_does_not_allow_test_prediction_reads():
    with pytest.raises(RuntimeError, match='prohibited'):
        guard().check(study.OUT / 'fold10_test_predictions.npz')

def test_native_cohort_selection_ignores_better_other_population_score():
    a = {'training_cohort': 'filtered', 'validation': {'filtered': {'macro_f1': .8, 'macro_auroc': .9},
                                                  'unfiltered': {'macro_f1': .6, 'macro_auroc': .8}}}
    b = {'training_cohort': 'filtered', 'validation': {'filtered': {'macro_f1': .7, 'macro_auroc': .9},
                                                  'unfiltered': {'macro_f1': .99, 'macro_auroc': .99}}}
    assert max([a, b], key=study.native_rank) is a

def test_unet_predict_rejects_fold10_before_checkpoint_access():
    with pytest.raises(AssertionError, match='fold9'):
        study.predict(Path('does_not_exist.pt'), pd.DataFrame({'strat_fold': [10]}))
