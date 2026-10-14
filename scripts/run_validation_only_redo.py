"""Redo test-feedback-affected choices using development data only.

Teacher weights are reused from the pre-ensemble frozen pool; six CNN students
are trained from scratch. No fold-10 evaluation is offered by this entrypoint.
Existing artifacts are protected, and historical test information is not read.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, re, runpy, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'src').exists(): ROOT=Path('D:/MyProject/college/ecg1')
sys.path.insert(0,str(ROOT))
from src.data.ptbxl import SUPERCLASSES
from src.models import build_model
from src.signal_processing.filters import FilterConfig
from src.training.data import load_labeled_splits, ensure_preprocessed_cache, CachedPTBXLDataset
from src.training.metrics import calculate_multilabel_metrics, optimize_thresholds, constrained_pr_thresholds
from src.training.train import train_baseline
from torch.utils.data import DataLoader

OUT=ROOT/'outputs/validation_only_redo_20261010'
CLASSES=list(SUPERCLASSES)

def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    temporary=path.with_suffix('.saving')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)
def status(message):
    print(message,flush=True);save(OUT/'status.json',{'stage':message,'time':datetime.now().isoformat()})

class Fold10ReadGuard:
    """Reject forbidden waveform and historical test-result reads, log inputs."""
    def __init__(self, metadata):
        root=str(Path('D:/datasets/ptb-xl').resolve()).replace('\\','/').lower()
        self.forbidden_records={root+'/'+str(v).lower()+suffix for v in metadata.loc[metadata.strat_fold.eq(10),'filename_lr'] for suffix in ('.dat','.hea')}
        self.forbidden_records.update(root+'/'+str(v).lower()+suffix for v in metadata.loc[metadata.strat_fold.eq(10),'filename_hr'] for suffix in ('.dat','.hea') if 'filename_hr' in metadata)
        self.reads=set();self.blocked=[]
    def check(self,path,mode='r'):
        if not isinstance(path,(str,bytes,os.PathLike)): return
        path=str(Path(os.fsdecode(path)).resolve()).replace('\\','/').lower()
        if isinstance(mode,str) and ('w' in mode or 'a' in mode or 'x' in mode) and '+' not in mode: return
        name=path.rsplit('/',1)[-1]
        forbidden=(path in self.forbidden_records or
                   ('/outputs/' in path and re.search(r'(^|_)test($|[_.])',name)) or
                   ('/outputs/' in path and name=='summary.json' and '/validation_only_redo_20261010/' not in path) or
                   ('/outputs/' in path and name in ['current_best.json','final_test.json']))
        if forbidden:
            self.blocked.append(path)
            raise RuntimeError('Fold10/historical test-result read prohibited: '+path)
        if '/outputs/' in path or '/datasets/ptb-xl/' in path: self.reads.add(path)
    def hook(self,event,args):
        if event=='open': self.check(args[0],args[1])

def metric(y,p,thresholds):
    m=calculate_multilabel_metrics(y,p,thresholds)
    ap=average_precision_score(y,p,average=None)
    m['macro_auprc']=float(ap.mean())
    for i,c in enumerate(CLASSES):
        m['per_class'][c]['auprc']=float(ap[i])
        # Independent scalar F1 and confusion-count checks.
        pred=np.asarray(p,dtype=np.float64)[:,i]>=thresholds[i]
        assert abs(f1_score(y[:,i],pred,zero_division=0)-m['per_class'][c]['f1'])<1e-12
        tp=int((pred&y[:,i].astype(bool)).sum());fp=int((pred&~y[:,i].astype(bool)).sum())
        fn=int((~pred&y[:,i].astype(bool)).sum());tn=len(y)-tp-fp-fn
        assert m['per_class'][c]['confusion_matrix']==[[tn,fp],[fn,tp]]
    return m

def fine_thresholds(y,p):
    grid=np.round(np.arange(10,91)/100,2)
    # Retain legacy cutoff convention at shared coarse-grid points.
    grid[::5]=np.arange(.1,.901,.05)
    p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=bool)
    pred=p[:,:,None]>=grid[None,None,:]
    tp=(pred&y[:,:,None]).sum(0);fp=(pred&~y[:,:,None]).sum(0);fn=(~pred&y[:,:,None]).sum(0)
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0)
    thresholds=[]
    for i in range(5):
        options=grid[np.isclose(f1[i],f1[i].max())]
        thresholds.append(float(options[np.argmin(abs(options-.5))]))
    return thresholds

def predict(cp,frame,split,logits=False):
    if not set(frame.strat_fold).issubset(set(range(1,9)) if split=='train' else {9}):
        raise ValueError('Prediction split must be development training or fold9.')
    cfg=cp['config']
    cache=ensure_preprocessed_cache(frame,cfg['dataset']['root'],ROOT/cfg['outputs']['cache_dir'],split_name=split+'_full',sampling_rate_hz=100,config=FilterConfig.from_mapping(cfg['signal_processing']))
    loader=DataLoader(CachedPTBXLDataset(cache,frame),batch_size=32,shuffle=False)
    model=build_model(cp['model_type'],dropout=cfg['model']['dropout']).to('cuda').eval()
    model.load_state_dict(cp['model_state'])
    values=[]
    with torch.inference_mode():
        for x,_ in loader:
            out=model(x.to('cuda'))
            values.append((out if logits else out.sigmoid()).cpu().numpy())
    del model
    return np.concatenate(values)

def rank(row): return (row['validation']['macro_f1'],row['validation']['macro_auroc'])

def teacher_selection(frames):
    parent=read(ROOT/'outputs/teacher_preprocessing/frozen_selection.json')
    rows=[]
    for original in parent['all_validation_results']:
        name=original['name'];path=Path(original['checkpoint'])
        cp=torch.load(path,map_location='cpu',weights_only=True)
        scores=predict(cp,frames['validation'],'validation')
        truth=frames['validation'][CLASSES].to_numpy(dtype=np.uint8)
        thresholds=optimize_thresholds(truth,scores)
        m=metric(truth,scores,thresholds)
        for key in ['macro_f1','macro_auroc']: assert abs(m[key]-original['validation'][key])<1e-10
        row={key:original[key] for key in ['name','model','preprocessing','seed','checkpoint']}
        row.update(checkpoint_sha256=digest(path),thresholds=thresholds,validation=m)
        rows.append(row)
        np.savez_compressed(OUT/'teacher_selection'/f'{name}_validation.npz',ecg_ids=frames['validation'].ecg_id,targets=truth,probabilities=scores,thresholds=thresholds)
    preprocessing=max([r for r in rows if r['model']=='inception1d' and r['seed']==42],key=rank)['preprocessing']
    family=max([r for r in rows if r['model']!='inception1d' and r['seed']==42 and r['preprocessing'] in {'P0_current',preprocessing}],key=rank)
    chosen=max([r for r in rows if r['model']==family['model'] and r['preprocessing']==family['preprocessing']],key=rank)
    assert chosen['name']==parent['selected_teacher']['name']
    frozen={'selected':chosen,'all_validation_results':rows,'preprocessing':preprocessing,'candidate_pool':'parent pre-ensemble frozen single models only','fold10_read':False,'frozen_at':datetime.now().isoformat()}
    save(OUT/'teacher_selection/frozen_selection.json',frozen)
    cp=torch.load(chosen['checkpoint'],map_location='cpu',weights_only=True)
    cp.update(thresholds=chosen['thresholds'],validation_metrics_tuned=chosen['validation'],correction_role='pre-ensemble teacher weights; fold9 selection replay')
    path=OUT/'teacher_selection/selected_single_teacher.pt';torch.save(cp,path)
    logits=predict(cp,frames['train'],'train',logits=True)
    np.savez_compressed(OUT/'teacher_training_logits.npz',ecg_ids=frames['train'].ecg_id,logits=logits,teacher_sha256=digest(path),superclasses=CLASSES)
    status('Pre-ensemble teacher selection replay complete: '+chosen['name'])
    return frozen

def redo_joint_thresholds(full_frames):
    folder=OUT/'joint_thresholds';folder.mkdir(exist_ok=True)
    frozen=read(ROOT/'outputs/further_optimization/final_selection.json')
    floors=frozen['threshold_policy']
    with np.load(ROOT/'outputs/further_optimization/screen/ensemble_equal/validation_predictions.npz') as a:
        np.testing.assert_array_equal(a['ecg_ids'],full_frames['validation'].ecg_id)
        np.testing.assert_array_equal(a['targets'],full_frames['validation'][CLASSES])
        thresholds,solver=constrained_pr_thresholds(a['targets'],a['probabilities'],precision_floor=floors['precision_floor'],recall_floor=floors['recall_floor'])
        assert thresholds is not None and solver['optimal']
        m=metric(a['targets'],a['probabilities'],thresholds)
    result={'thresholds':thresholds,'solver':solver,'validation':m,'floor_source':'pre-first-ensemble-test final_selection.json, fold9 only','precision_floor':floors['precision_floor'],'recall_floor':floors['recall_floor'],'fold10_evaluation':None,'cv':[]}
    save(folder/'frozen_full_validation.json',result)
    # Repeat the pre-test internal CV model choice from development folds.
    cases=read(ROOT/'outputs/further_optimization/screen_selection.json')['cv_cases']
    model_checks={}
    for case in cases:
        candidate_scores=[];reference_scores=[]
        for fold in range(1,5):
            directory=ROOT/f'outputs/further_optimization/cv/fold_{fold}/{case}'
            refdir=ROOT/f'outputs/cross_validation/fold_{fold}/resnet_global'
            with np.load(refdir/'calibration_predictions.npz') as a:
                refthr=fine_or_pr_reference(a)
                refcal=metric(a['targets'],a['probabilities'],refthr)
            with np.load(directory/'calibration_predictions.npz') as a:
                # Original main CV criterion uses exact PR F1, not 0.01.
                from scripts.run_optimization import exact_thresholds
                mainthr=exact_thresholds(a['targets'],a['probabilities'],[.5]*5)
            with np.load(directory/'heldout_predictions.npz') as a:
                check_internal_ids(a,full_frames)
                candidate_scores.append(metric(a['targets'],a['probabilities'],mainthr))
            with np.load(refdir/'heldout_predictions.npz') as a:
                check_internal_ids(a,full_frames)
                reference_scores.append(metric(a['targets'],a['probabilities'],refthr))
            if case=='ensemble_equal':
                with np.load(directory/'calibration_predictions.npz') as a:
                    th,s=constrained_pr_thresholds(a['targets'],a['probabilities'],precision_floor=refcal['macro_precision'],recall_floor=refcal['macro_recall'])
                    assert th is not None and s['optimal']
                    cal=metric(a['targets'],a['probabilities'],th)
                save(folder/f'cv{fold}_frozen_thresholds.json',{'thresholds':th,'calibration':cal,'solver':s})
                with np.load(directory/'heldout_predictions.npz') as a:
                    held=metric(a['targets'],a['probabilities'],th)
                result['cv'].append({'fold':fold,'thresholds':th,'calibration':cal,'heldout_training_pool':held,'solver':s})
        f1=float(np.mean([m['macro_f1'] for m in candidate_scores]));rf1=float(np.mean([m['macro_f1'] for m in reference_scores]))
        auc=float(np.mean([m['macro_auroc'] for m in candidate_scores]));rauc=float(np.mean([m['macro_auroc'] for m in reference_scores]))
        wins=sum(a['macro_f1']>b['macro_f1'] for a,b in zip(candidate_scores,reference_scores))
        model_checks[case]={'mean_f1':f1,'reference_mean_f1':rf1,'wins':wins,'eligible':f1>=rf1+.001 and wins>=3 and auc>=rauc-.001}
    eligible=[(v['mean_f1'],name) for name,v in model_checks.items() if v['eligible']]
    result['cv_model_choice']=max(eligible)[1] if eligible else 'retain_previous'
    result['cv_model_checks']=model_checks
    result['cv_mean_macro_f1']=float(np.mean([r['heldout_training_pool']['macro_f1'] for r in result['cv']]))
    save(folder/'result.json',result)
    status('Joint threshold and internal CV choice redo complete; no fold10 evaluation')
    return result

def fine_or_pr_reference(a):
    from scripts.run_optimization import exact_thresholds
    return exact_thresholds(a['targets'],a['probabilities'],[.5]*5)
def check_internal_ids(a,frames):
    reference=frames['train'].set_index('ecg_id').loc[a['ecg_ids']]
    assert reference.strat_fold.le(8).all()
    np.testing.assert_array_equal(a['targets'],reference[CLASSES])

def train_students(protocol,frames):
    rows=[]
    teacher=OUT/'teacher_selection/selected_single_teacher.pt'
    logits=OUT/'teacher_training_logits.npz'
    for seed in [42,43,44]:
        for arm in ['supervised','single_teacher_kd']:
            name=f'{arm}_seed{seed}';folder=OUT/name
            cfg=copy.deepcopy(protocol['student_config'])
            cfg['training']['random_seed']=seed
            if arm=='single_teacher_kd': cfg['training']['distillation']={'alpha':.5,'temperature':2.,'logits_path':str(logits),'teacher_checkpoint':str(teacher),'teacher_sha256':digest(teacher)}
            for key,sub in [('checkpoint_dir','checkpoint'),('metrics_dir','metrics'),('figure_dir','figure'),('predictions_dir','predictions')]:cfg['outputs'][key]=str((folder/sub).relative_to(ROOT))
            cfgpath=folder/'config.json'
            if cfgpath.exists(): assert read(cfgpath)==cfg
            else:save(cfgpath,cfg)
            path=folder/'checkpoint/cnn1d_best.pt';history=folder/'metrics/training_history_cnn1d_full.json'
            if path.exists() and not history.exists(): raise RuntimeError('Incomplete run; inspect before resuming: '+name)
            if not history.exists():
                status('Fresh CNN training: '+name)
                train_baseline(cfg,project_root=ROOT,model_type='cnn1d')
            cp=torch.load(path,map_location='cpu',weights_only=True)
            assert cp['training_records']==17084 and cp['validation_records']==2146
            p=predict(cp,frames['validation'],'validation');y=frames['validation'][CLASSES].to_numpy(dtype=np.uint8)
            threshold=fine_thresholds(y,p);m=metric(y,p,threshold)
            cp.update(thresholds=threshold,validation_metrics_tuned=m,threshold_search='fold9_only_0.01_grid',correction_role='fresh student; no fold10 evaluation')
            torch.save(cp,path)
            np.savez_compressed(folder/'validation_predictions.npz',ecg_ids=frames['validation'].ecg_id,targets=y,probabilities=p,thresholds=threshold)
            row={'name':name,'arm':arm,'seed':seed,'checkpoint':str(path),'checkpoint_sha256':digest(path),'epoch':cp['epoch'],'thresholds':threshold,'validation':m,'epochs_completed':len(read(history)['history'])}
            save(folder/'result.json',row);rows.append(row)
    chosen=max(rows,key=rank)
    save(OUT/'frozen_student_selection.json',{'selected':chosen,'all_candidates':rows,'criterion':'fold9 FP32 fine-grid macro F1, then AUROC','fold10_evaluated':False,'frozen_at':datetime.now().isoformat()})
    status('Six fresh students complete; validation-selected '+chosen['name'])
    return rows,chosen

def invoke(script,args):
    previous=sys.argv;sys.argv=[str(ROOT/script),*map(str,args)]
    try:runpy.run_path(str(ROOT/script),run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (0,None):raise
    finally:sys.argv=previous

def export_selection(chosen,frames):
    folder=OUT/'selected_export';folder.mkdir(exist_ok=True)
    if not (folder/'deployment_manifest.json').exists():invoke('scripts/prepare_stm32.py',['--checkpoint',chosen['checkpoint'],'--output-dir',folder])
    if not (folder/'int8_validation_report.json').exists():invoke('scripts/quantize_cubeai.py',['--export-dir',folder])
    manifest=read(folder/'deployment_manifest.json')
    assert set(manifest['calibration']['ecg_ids'])<=set(frames['train'].ecg_id)
    candidates=[]
    for name,source in [('original_seed44_int8',ROOT/'outputs/distillation/distilled_export'),(chosen['name']+'_int8',folder)]:
        with np.load(source/'int8_validation_predictions.npz') as a:
            np.testing.assert_array_equal(a['ecg_id'],frames['validation'].ecg_id)
            np.testing.assert_array_equal(a['targets'],frames['validation'][CLASSES])
            th=fine_thresholds(a['targets'],a['probabilities_int8'])
            m=metric(a['targets'],a['probabilities_int8'],th)
        candidates.append({'name':name,'onnx':str(source/'ptbxl_cnn1d_int8.onnx'),'onnx_sha256':digest(source/'ptbxl_cnn1d_int8.onnx'),'thresholds':th,'validation':m,'firmware_generated':False})
    selected=max(candidates,key=rank)
    result={'selected':selected,'candidates':candidates,'fold10_evaluation':None,'physical_board_test':False,'criterion':'fold9 INT8 macro F1 then AUROC; each model own fine thresholds','frozen_at':datetime.now().isoformat()}
    save(OUT/'frozen_deployment_selection.json',result)
    save(folder/'candidate_thresholds.json',{'labels':CLASSES,'onnx_sha256':candidates[1]['onnx_sha256'],'thresholds':candidates[1]['thresholds'],'threshold_source':'fold9','historical_firmware_modified':False})
    status('Training-only INT8 calibration and validation-only deployment choice complete')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['all','prepare','train','export'],default='all');args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'teacher_selection').mkdir(exist_ok=True)
    assert torch.cuda.is_available(),'Fresh training requires the existing CUDA environment.'
    torch.set_num_threads(8)
    full=load_labeled_splits('D:/datasets/ptb-xl',100)
    allmeta=pd.concat(list(full.values())).sort_index()
    guard=Fold10ReadGuard(allmeta);sys.addaudithook(guard.hook)
    frames={s:f[f[CLASSES].sum(axis=1).gt(0)].copy() for s,f in full.items() if s in ['train','validation']}
    assert len(frames['train'])==17084 and len(frames['validation'])==2146
    assert not set(frames['train'].patient_id)&set(frames['validation'].patient_id)
    base=torch.load(ROOT/'outputs/distillation/baseline_seed42/checkpoint/cnn1d_best.pt',map_location='cpu',weights_only=True)['config']
    base=copy.deepcopy(base);base['training'].pop('distillation',None)
    base['signal_processing'].pop('training_mean',None);base['signal_processing'].pop('training_std',None)
    assert base['dataset']['train_folds']==list(range(1,9)) and base['dataset']['validation_fold']==9
    paths=[ROOT/'outputs/distillation/distilled_seed44/checkpoint/cnn1d_best.pt',ROOT/'outputs/teacher_distillation/ensemble_kd_seed42/checkpoint/cnn1d_best.pt',ROOT/'outputs/teacher_preprocessing/ensemble_followup/recommended_teacher.pt',ROOT/'outputs/distillation/distilled_export/ptbxl_cnn1d_int8.onnx',ROOT/'outputs/teacher_distillation/selected_export/ptbxl_cnn1d_int8.onnx',ROOT/'outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_STM32F407VE.hex']
    parent=read(ROOT/'outputs/teacher_preprocessing/frozen_selection.json')
    paths.extend(Path(row['checkpoint']) for row in parent['all_validation_results'])
    protocol={'date':'2026-10-10','scope':['joint-threshold and internal-CV selection redo','pre-ensemble single-teacher selection replay','six freshly initialized CNN training runs','validation-only INT8 comparison'], 'teacher_weights_retrained':False,'teacher_pool_source':'parent frozen_selection, before ensemble follow-up','student_seeds':[42,43,44],'student_config':base,'teacher_threshold_grid':'original validation coarse 0.05','student_threshold_grid':'fixed validation grid 0.10-0.90, step0.01; tie nearest0.5','model_choice':'FP32 validation macro F1 then AUROC; ordinary and KD students both eligible','checkpoint_choice':'validation AUROC, same original training code','test_waveforms_or_saved_predictions_read':False,'historical_test_influence_erased':False,'new_blind_test_claim':False,'independent_generalization_status':'unavailable; prior test knowledge persists','protected_artifacts':{str(p):digest(p) for p in paths},'source_sha256':digest(Path(__file__))}
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol,'Changed protocol; use a new directory.'
    else:save(OUT/'protocol.json',protocol)
    status('Protocol frozen before new inference/training; fold10 read guard active')
    if not (OUT/'teacher_selection/frozen_selection.json').exists():teacher_selection(frames)
    if not (OUT/'joint_thresholds/result.json').exists():redo_joint_thresholds(full)
    if args.phase=='prepare':return
    rows,chosen=train_students(protocol,frames)
    if args.phase=='train':return
    deployment=export_selection(chosen,frames)
    for path,expected in protocol['protected_artifacts'].items():assert digest(path)==expected,path
    assert not guard.blocked
    summary={'protocol_sha256':digest(OUT/'protocol.json'),'teacher':read(OUT/'teacher_selection/frozen_selection.json')['selected'],'joint_thresholds':read(OUT/'joint_thresholds/result.json'),'students':rows,'selected_student':chosen,'deployment':deployment,'fold10_evaluation':None,'protected_artifacts_unchanged':True,'completed_at':datetime.now().isoformat()}
    save(OUT/'summary.json',summary)
    save(OUT/'audit.json',{'passed':True,'fold10_waveform_reads':0,'historical_test_result_reads':0,'forbidden_read_attempts':guard.blocked,'development_file_reads':sorted(guard.reads),'protected_artifacts_unchanged':True,'student_runs_fresh':6,'teacher_weights_reused':True,'student_and_int8_metrics_independently_checked':True})
    status('Validation-only redo complete; six fresh students, independent checks, no fold10 result')

if __name__=='__main__':main()
