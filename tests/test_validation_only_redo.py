import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/run_validation_only_redo.py'
spec=importlib.util.spec_from_file_location('validation_only_redo',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def guard():
    return module.Fold10ReadGuard(pd.DataFrame({'strat_fold':[1,10],'filename_lr':['records100/00000/00001_lr','records100/00000/00009_lr'],'filename_hr':['records500/00000/00001_hr','records500/00000/00009_hr']}))

@pytest.mark.parametrize('path',[
    'D:/datasets/ptb-xl/records100/00000/00009_lr.dat',
    'D:/datasets/ptb-xl/records500/00000/00009_hr.hea',
    'D:/MyProject/college/ecg1/outputs/distillation/int8_test_predictions.npz',
    'D:/MyProject/college/ecg1/outputs/optimization/final_test.json',
    'D:/MyProject/college/ecg1/outputs/teacher_preprocessing/summary.json',
])
def test_guard_rejects_fold10_and_mixed_historical_results(path):
    with pytest.raises(RuntimeError,match='prohibited'):guard().check(path)

@pytest.mark.parametrize('path',[
    'D:/datasets/ptb-xl/records100/00000/00001_lr.dat',
    'D:/MyProject/college/ecg1/outputs/cache/train_full_abc.npy',
    'D:/MyProject/college/ecg1/outputs/teacher_preprocessing/frozen_selection.json',
    'D:/MyProject/college/ecg1/outputs/further_optimization/cv/fold_1/ensemble_equal/heldout_predictions.npz',
    'D:/MyProject/college/ecg1/outputs/validation_only_redo_20261010/summary.json',
])
def test_guard_allows_training_validation_and_corrected_outputs(path):
    guard().check(path)

def test_selection_uses_validation_even_when_test_ranking_reversed():
    rows=[{'name':'validation_winner','validation':{'macro_f1':.8,'macro_auroc':.9},'test':{'macro_f1':.1}},
          {'name':'test_winner','validation':{'macro_f1':.7,'macro_auroc':.95},'test':{'macro_f1':.99}}]
    assert max(rows,key=module.rank)['name']=='validation_winner'
    for row in rows:row.pop('test')
    assert max(rows,key=module.rank)['name']=='validation_winner'

def test_predict_rejects_fold10_before_model_or_cache_access():
    with pytest.raises(ValueError,match='fold9'):
        module.predict({},pd.DataFrame({'strat_fold':[10]}),'validation')

def test_threshold_search_matches_exhaustive_validation_f1():
    from sklearn.metrics import f1_score
    rng=np.random.default_rng(42);y=rng.integers(0,2,size=(60,5));p=rng.random((60,5))
    th=module.fine_thresholds(y,p)
    grid=np.round(np.arange(10,91)/100,2);grid[::5]=np.arange(.1,.901,.05)
    for i in range(5):
        assert abs(f1_score(y[:,i],p[:,i]>=th[i])-max(f1_score(y[:,i],p[:,i]>=t) for t in grid))<1e-12
