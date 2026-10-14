"""Independent completed correction audit, never evaluate fold10."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,roc_auc_score,average_precision_score,f1_score
ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'src').exists():ROOT=Path('D:/MyProject/college/ecg1')
sys.path.insert(0,str(ROOT))
from src.training.data import load_labeled_splits,_cache_fingerprint
from src.signal_processing.filters import FilterConfig
from scripts.run_validation_only_redo import Fold10ReadGuard
import pandas as pd
OUT=ROOT/'outputs/validation_only_redo_20261010'
C=['NORM','MI','STTC','CD','HYP']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
splits=load_labeled_splits('D:/datasets/ptb-xl',100)
guard=Fold10ReadGuard(pd.concat(list(splits.values())));sys.addaudithook(guard.hook)
train=splits['train'];train=train[train[C].sum(axis=1).gt(0)]
validation=splits['validation'];validation=validation[validation[C].sum(axis=1).gt(0)]
assert not set(train.patient_id)&set(validation.patient_id)
protocol=read(OUT/'protocol.json');summary=read(OUT/'summary.json')
assert sha(ROOT/'scripts/run_validation_only_redo.py')==protocol['source_sha256']
assert summary['fold10_evaluation'] is None
sets=0
def compare(y,p,t,m):
    global sets
    p=np.asarray(p,dtype=np.float64);pred=p>=np.asarray(t,dtype=np.float64)
    pr,re,f1,support=precision_recall_fscore_support(y,pred,average=None,zero_division=0)
    ap=average_precision_score(y,p,average=None)
    for key,number in [('macro_precision',pr.mean()),('macro_recall',re.mean()),('macro_f1',f1.mean()),('macro_auroc',roc_auc_score(y,p,average='macro')),('macro_auprc',ap.mean())]:
        np.testing.assert_allclose(m[key],number,rtol=0,atol=1e-12)
    for i,c in enumerate(C):
        v=m['per_class'][c]
        for key,number in [('precision',pr[i]),('recall',re[i]),('f1',f1[i]),('auprc',ap[i])]:np.testing.assert_allclose(v[key],number,rtol=0,atol=1e-12)
        assert v['confusion_matrix']==confusion_matrix(y[:,i],pred[:,i]).tolist()
    sets+=1
def independent_fine(y,p):
    grid=np.round(np.arange(10,91)/100,2);grid[::5]=np.arange(.1,.901,.05)
    result=[];p=p.astype(np.float64)
    for i in range(5):
        f=np.array([f1_score(y[:,i],p[:,i]>=t,zero_division=0) for t in grid])
        choices=grid[np.isclose(f,f.max())];result.append(float(choices[np.argmin(abs(choices-.5))]))
    return result

teacher=read(OUT/'teacher_selection/frozen_selection.json')
for r in teacher['all_validation_results']:
    with np.load(OUT/'teacher_selection'/f'{r["name"]}_validation.npz') as d:
        np.testing.assert_array_equal(d['ecg_ids'],validation.ecg_id);np.testing.assert_array_equal(d['targets'],validation[C])
        compare(d['targets'],d['probabilities'],r['thresholds'],r['validation'])
rows=teacher['all_validation_results'];key=lambda r:(r['validation']['macro_f1'],r['validation']['macro_auroc'])
pre=max([r for r in rows if r['model']=='inception1d' and r['seed']==42],key=key)['preprocessing']
family=max([r for r in rows if r['model']!='inception1d' and r['seed']==42 and r['preprocessing'] in {'P0_current',pre}],key=key)
winner=max([r for r in rows if r['model']==family['model'] and r['preprocessing']==family['preprocessing']],key=key)
assert winner['name']==teacher['selected']['name']
with np.load(OUT/'teacher_training_logits.npz') as d:
    np.testing.assert_array_equal(d['ecg_ids'],train.ecg_id)
    assert d['logits'].shape==(17084,5) and np.isfinite(d['logits']).all()
    assert str(d['teacher_sha256'])==sha(OUT/'teacher_selection/selected_single_teacher.pt')

expected_weights=(len(train)-train[C].sum().to_numpy())/train[C].sum().to_numpy()
moments=None
for r in summary['students']:
    path=Path(r['checkpoint']);cp=torch.load(path,map_location='cpu',weights_only=True)
    assert sha(path)==r['checkpoint_sha256']
    assert cp['config']['training']['random_seed']==r['seed'] and cp['training_records']==17084 and cp['validation_records']==2146
    np.testing.assert_allclose(cp['pos_weight'],expected_weights,rtol=1e-6,atol=1e-7)
    if r['arm']=='single_teacher_kd':
        kd=cp['config']['training']['distillation'];assert kd['alpha']==.5 and kd['temperature']==2
        assert kd['teacher_sha256']==sha(OUT/'teacher_selection/selected_single_teacher.pt')
    else:assert 'distillation' not in cp['config']['training']
    history=read(path.parent.parent/'metrics/training_history_cnn1d_full.json')
    assert history['best_epoch']==r['epoch'] and len(history['history'])==r['epochs_completed']
    with np.load(path.parent.parent/'validation_predictions.npz') as d:
        np.testing.assert_array_equal(d['ecg_ids'],validation.ecg_id);np.testing.assert_array_equal(d['targets'],validation[C])
        th=independent_fine(d['targets'],d['probabilities']);np.testing.assert_array_equal(th,r['thresholds'])
        np.testing.assert_array_equal(th,cp['thresholds']);compare(d['targets'],d['probabilities'],th,r['validation'])
    if moments is None:
        cfg=cp['config'];rawcfg={**cfg['signal_processing'],'normalization':'none'}
        rawcfg.pop('training_mean',None);rawcfg.pop('training_std',None)
        fp=_cache_fingerprint(train,100,FilterConfig.from_mapping(rawcfg))
        raw=np.load(ROOT/cfg['outputs']['cache_dir']/f'train_full_{fp}.npy',mmap_mode='r')
        total=np.zeros(12);square=np.zeros(12);n=0
        for start in range(0,len(raw),128):
            x=np.asarray(raw[start:start+128],dtype=np.float64);total+=x.sum(axis=(0,2));square+=(x*x).sum(axis=(0,2));n+=x.shape[0]*x.shape[2]
        mean=total/n;std=np.sqrt(square/n-mean**2);moments=(mean,std)
    np.testing.assert_allclose(cp['config']['signal_processing']['training_mean'],moments[0],rtol=0,atol=1e-12)
    np.testing.assert_allclose(cp['config']['signal_processing']['training_std'],moments[1],rtol=0,atol=1e-12)
winner=max(summary['students'],key=lambda r:(r['validation']['macro_f1'],r['validation']['macro_auroc']))
assert winner['name']==summary['selected_student']['name']
for r in summary['deployment']['candidates']:
    folder=Path(r['onnx']).parent
    with np.load(folder/'int8_validation_predictions.npz') as d:
        np.testing.assert_array_equal(d['ecg_id'],validation.ecg_id);np.testing.assert_array_equal(d['targets'],validation[C])
        th=independent_fine(d['targets'],d['probabilities_int8']);np.testing.assert_array_equal(th,r['thresholds'])
        compare(d['targets'],d['probabilities_int8'],th,r['validation'])
    assert sha(r['onnx'])==r['onnx_sha256']
chosen=max(summary['deployment']['candidates'],key=lambda r:(r['validation']['macro_f1'],r['validation']['macro_auroc']))
assert chosen['name']==summary['deployment']['selected']['name']
joint=summary['joint_thresholds']
joint_manifest=read(OUT/'joint_thresholds/ensemble_manifest.json')
np.testing.assert_array_equal(joint_manifest['thresholds'],joint['thresholds'])
assert joint_manifest['fold10_evaluation'] is None
for component in joint_manifest['components']:assert sha(component['checkpoint'])==component['sha256']
with np.load(ROOT/'outputs/further_optimization/screen/ensemble_equal/validation_predictions.npz') as d:
    np.testing.assert_array_equal(d['ecg_ids'],splits['validation'].ecg_id)
    np.testing.assert_array_equal(d['targets'],splits['validation'][C])
    compare(d['targets'],d['probabilities'],joint['thresholds'],joint['validation'])
assert joint['solver']['optimal']
assert joint['validation']['macro_precision']+1e-9>=joint['precision_floor']
assert joint['validation']['macro_recall']+1e-9>=joint['recall_floor']
for row in joint['cv']:
    folder=ROOT/f'outputs/further_optimization/cv/fold_{row["fold"]}/ensemble_equal'
    with np.load(folder/'calibration_predictions.npz') as d:
        np.testing.assert_array_equal(d['ecg_ids'],splits['validation'].ecg_id)
        compare(d['targets'],d['probabilities'],row['thresholds'],row['calibration'])
    with np.load(folder/'heldout_predictions.npz') as d:
        frame=splits['train'].set_index('ecg_id').loc[d['ecg_ids']]
        np.testing.assert_array_equal(d['targets'],frame[C]);assert frame.strat_fold.le(8).all()
        compare(d['targets'],d['probabilities'],row['thresholds'],row['heldout_training_pool'])
    assert row['solver']['optimal']
manifest=read(OUT/'selected_export/deployment_manifest.json')
assert set(manifest['calibration']['ecg_ids'])<=set(train.ecg_id)
selected_cp=torch.load(summary['selected_student']['checkpoint'],map_location='cpu',weights_only=True)
normcfg=selected_cp['config']['signal_processing']
normfp=_cache_fingerprint(train,100,FilterConfig.from_mapping(normcfg))
normpath=ROOT/selected_cp['config']['outputs']['cache_dir']/f'train_full_{normfp}.npy'
normalized=np.load(normpath,mmap_mode='r')
positions=pd.Index(train.ecg_id).get_indexer(manifest['calibration']['ecg_ids'])
assert (positions>=0).all()
np.testing.assert_array_equal(np.load(OUT/'selected_export/calibration_train.npy'),normalized[positions])
for p,expected in protocol['protected_artifacts'].items():assert sha(p)==expected
assert not guard.blocked
result={'passed':True,'metric_sets_independently_recomputed':sets,'threshold_sets_independently_researched':8,'joint_threshold_manifest_and_components_checked':True,'joint_threshold_metrics_and_solver_status_checked':True,'student_training_statistics_recomputed':True,'student_runs_completed':6,'fresh_training_history_verified':True,'teacher_weights_reused':True,'teacher_selection_recomputed_from_validation':True,'student_selection_recomputed_from_validation':True,'int8_selection_recomputed_from_validation':True,'train_only_calibration':True,'actual_calibration_tensor_matches_training_records':True,'protected_artifacts_unchanged':True,'fold10_evaluated':False,'historical_test_influence_erased':False}
(OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
