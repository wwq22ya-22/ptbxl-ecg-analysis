"""1D U-Net classification study with filtered/unfiltered training, fold9 only.

Two architectures x two training cohorts x three paired seeds = 12 fresh runs.
No fold10 inference/evaluation stage. Earlier teacher weights are reused only
as fixed validation references; deployment artifacts are never replaced.
"""
from __future__ import annotations
import argparse,copy,json,hashlib,sys,time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'src').exists():ROOT=Path('D:/MyProject/college/ecg1')
sys.path.insert(0,str(ROOT))
from src.data.ptbxl import SUPERCLASSES
from src.models import build_model
from src.training.data import load_labeled_splits,ensure_preprocessed_cache,CachedPTBXLDataset
from src.training.train import train_baseline
from src.signal_processing.filters import FilterConfig
from scripts.run_validation_only_redo import Fold10ReadGuard,fine_thresholds,metric

OUT=ROOT/'outputs/unet_teacher_20261010'
C=list(SUPERCLASSES)
COHORTS=('filtered','unfiltered')
MODELS=('unet1d','unet_encoder1d')
SEEDS=(42,43,44)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.saving');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(path)
def status(message):
    print(message,flush=True);save(OUT/'status.json',{'time':datetime.now().isoformat(),'stage':message})


class UNetReadGuard(Fold10ReadGuard):
    def check(self,path,mode='r'):
        if isinstance(path,(str,bytes,Path)):
            actual=Path(path).resolve()
            if actual.name=='summary.json' and actual.is_relative_to(OUT):
                self.reads.add(str(actual));return
        super().check(path,mode)


def native_rank(row):
    m=row['validation'][row['training_cohort']]
    return m['macro_f1'],m['macro_auroc']


def make_config(base,cohort,model,seed):
    cfg=copy.deepcopy(base)
    cfg['dataset']['require_diagnostic_label']=cohort=='filtered'
    cfg['model']={'type':model,'dropout':.2}
    cfg['training'].update(epochs=30,early_stopping_patience=8,batch_size=32,
                           learning_rate=.001,weight_decay=.0001,random_seed=seed,use_amp=True,num_workers=0)
    for k in ['distillation','scheduler','mixup_alpha','augmentation_gain','augmentation_noise','pos_weight_power','loss']:
        cfg['training'].pop(k,None)
    for k in ['training_mean','training_std']:cfg['signal_processing'].pop(k,None)
    cfg['signal_processing']['normalization']='train_global'
    folder=OUT/cohort/f'{model}_seed{seed}'
    for key,name in [('checkpoint_dir','checkpoint'),('metrics_dir','metrics'),('figure_dir','figure'),('predictions_dir','predictions')]:
        cfg['outputs'][key]=str((folder/name).relative_to(ROOT))
    return cfg


def prepare(frames):
    protocol_path=OUT/'protocol.json'
    code={str(ROOT/p):sha(ROOT/p) for p in ['src/models/unet1d.py','src/models/__init__.py','scripts/run_unet_teacher.py','src/training/train.py','src/training/data.py','scripts/run_validation_only_redo.py']}
    parent=read(ROOT/'outputs/teacher_preprocessing/frozen_selection.json')
    references=[r for r in parent['all_validation_results'] if r['model']=='inception_time1d' and r['preprocessing']=='P0_current']
    assert {r['seed'] for r in references}==set(SEEDS)
    base=read(ROOT/'outputs/validation_only_redo_20261010/protocol.json')['student_config']
    configs={f'{cohort}/{model}_seed{seed}':make_config(base,cohort,model,seed) for cohort in COHORTS for model in MODELS for seed in SEEDS}
    if protocol_path.exists():
        protocol=read(protocol_path)
        assert protocol['code_sha256']==code and protocol['configs']==configs,'Frozen protocol/config/source mismatch'
        for path,digest in protocol['protected_artifacts'].items():assert sha(path)==digest,path
        return protocol
    protected=dict(read(ROOT/'outputs/validation_only_redo_20261010/protocol.json')['protected_artifacts'])
    for p in (ROOT/'outputs/validation_only_redo_20261010').rglob('*'):
        if p.suffix in ('.pt','.onnx'):protected[str(p)]=sha(p)
    for r in references:protected[r['checkpoint']]=sha(r['checkpoint'])
    for cohort in COHORTS:
        train=frames[cohort]['train'];val=frames[cohort]['validation']
        assert not set(train.patient_id)&set(val.patient_id)
        assert train.strat_fold.between(1,8).all() and val.strat_fold.eq(9).all()
        for name,frame in [('train',train),('validation',val)]:
            frame[['ecg_id','patient_id','strat_fold',*C]].to_csv(OUT/f'{cohort}_{name}_cohort.csv',index=False)
    protocol={'created_at':datetime.now().isoformat(),'scope':'fresh 1D U-Net classification and same-encoder ablation; two training cohorts; validation only',
              'classification_not_segmentation':True,'architectures':{'unet1d':{'parameters':579053,'widths':[12,24,48,96,192],'encoder_lengths':[1000,500,250,125,62],'decoder_lengths':[125,250,500,1000],'kernel':5,'up':'linear to exact skip length + 1x1','head':'global average+maximum pooling, dropout .2, 5 logits'},'unet_encoder1d':{'parameters':372053,'same_encoder':True,'decoder':False,'parameter_matched':False}},
              'seeds':list(SEEDS),'run_count':12,'configs':configs,'code_sha256':code,'protected_artifacts':protected,
              'reference_teachers':references,'epoch_selection':'fold9 macro AUROC improvement > 1e-4, early stopping 8',
              'candidate_selection':'within each training cohort and architecture: native fold9 FP32 fine-grid macro F1, then AUROC',
              'threshold_search':'each validation cohort separately: .10-.90 by .01; retain legacy float cutoffs at .05 multiples; F1 tie closest .5',
              'both_validation_cohorts_evaluated':True,'no_cross_cohort_ranking':True,'test_fold_evaluated':False,'existing_test_independence_restored':False,
              'cohorts':{cohort:{part:len(frames[cohort][part]) for part in ['train','validation']} for cohort in COHORTS},
              'restrictions':['no test-fold waveform/prediction/result reads','train-only normalization and positive loss weights','no augmentation, KD, tuning after test, quantization, firmware generation, or external validation','all 12 configurations fixed before training; negative results retained','encoder-only ablation changes decoder and head/parameter count together; cannot fully isolate parameter effects']}
    save(protocol_path,protocol)
    for name,cfg in configs.items():save(OUT/name/'config.json',cfg)
    status('Frozen 12-run protocol; no fold10 evaluation')
    return protocol


def predict(path,frame):
    assert frame.strat_fold.eq(9).all(),'Only fold9 inference is permitted'
    cp=torch.load(path,map_location='cpu',weights_only=True);cfg=cp['config']
    cache=ensure_preprocessed_cache(frame,cfg['dataset']['root'],ROOT/cfg['outputs']['cache_dir'],
                                   split_name='validation_full',sampling_rate_hz=100,config=FilterConfig.from_mapping(cfg['signal_processing']))
    loader=DataLoader(CachedPTBXLDataset(cache,frame),batch_size=32,shuffle=False)
    model=build_model(cp['model_type'],dropout=cfg['model']['dropout']).to('cuda').eval();model.load_state_dict(cp['model_state'])
    logits=[]
    with torch.inference_mode():
        for x,_ in loader:logits.append(model(x.to('cuda')).cpu().numpy())
    del model
    z=np.concatenate(logits);prob=torch.from_numpy(z).sigmoid().numpy()
    assert z.shape==(len(frame),5) and np.isfinite(z).all()
    return z,prob


def evaluate_row(path,frames,folder):
    evaluations={}
    for cohort in COHORTS:
        frame=frames[cohort]['validation'];z,p=predict(path,frame);y=frame[C].to_numpy(dtype=np.uint8)
        threshold=fine_thresholds(y,p);m=metric(y,p,threshold);evaluations[cohort]=m
        np.savez_compressed(folder/f'{cohort}_validation_predictions.npz',ecg_ids=frame.ecg_id.to_numpy(),targets=y,logits=z,probabilities=p,thresholds=np.asarray(threshold),checkpoint_sha256=np.asarray(sha(path)))
    return evaluations


def train_runs(protocol,frames):
    rows=[]
    # Paired models alternate; completing all three seeds is precommitted.
    for seed in SEEDS:
        for cohort in COHORTS:
            for model in MODELS:
                name=f'{cohort}/{model}_seed{seed}';folder=OUT/name;cfg=protocol['configs'][name]
                assert read(folder/'config.json')==cfg
                path=folder/'checkpoint'/f'{model}_best.pt'
                history=folder/'metrics'/f'training_history_{model}_full.json'
                result=folder/'result.json'
                if result.exists():
                    row=read(result);assert sha(path)==row['checkpoint_sha256'];rows.append(row);continue
                if path.exists() and not history.exists():raise RuntimeError('Partial run requires explicit review before resuming: '+name)
                if not history.exists():
                    status('Fresh training '+name);train_baseline(cfg,project_root=ROOT,model_type=model)
                cp=torch.load(path,map_location='cpu',weights_only=True)
                assert cp['training_records']==len(frames[cohort]['train']) and cp['validation_records']==len(frames[cohort]['validation'])
                evaluation=evaluate_row(path,frames,folder)
                h=read(history)
                row={'name':name,'model':model,'seed':seed,'training_cohort':cohort,'checkpoint':str(path),'checkpoint_sha256':sha(path),
                     'parameters':sum(p.numel() for p in build_model(model).parameters()),'best_epoch':cp['epoch'],'epochs_completed':len(h['history']),
                     'training_seconds':sum(e['seconds'] for e in h['history']),'train_records':cp['training_records'],'validation':evaluation}
                # Store fine thresholds separately; original checkpoint coarse calibration is retained.
                save(folder/'frozen_validation_thresholds.json',{'checkpoint_sha256':row['checkpoint_sha256'],'native_validation_cohort':cohort,'thresholds':{pop:[m['per_class'][c]['threshold'] for c in C] for pop,m in evaluation.items()},'source':'fold9 only'})
                save(result,row);rows.append(row)
                status(f"Completed {name}; native val F1={evaluation[cohort]['macro_f1']:.6f}")
    return rows


def references(protocol,frames):
    rows=[]
    for r in protocol['reference_teachers']:
        folder=OUT/'references'/r['name'];folder.mkdir(parents=True,exist_ok=True)
        path=Path(r['checkpoint']);assert sha(path)==protocol['protected_artifacts'][str(path)]
        m=evaluate_row(path,frames,folder)
        rows.append({'name':r['name'],'model':r['model'],'seed':r['seed'],'training_cohort':'filtered','checkpoint':str(path),'checkpoint_sha256':sha(path),'validation':m,'weights_retrained':False})
    return rows


def summarize(rows,ref):
    selected={};statistics={};paired={}
    for cohort in COHORTS:
        selected[cohort]={};statistics[cohort]={};paired[cohort]={}
        for model in MODELS:
            candidates=[r for r in rows if r['training_cohort']==cohort and r['model']==model]
            selected[cohort][model]=max(candidates,key=native_rank)
            statistics[cohort][model]={}
            for population in COHORTS:
                vals=[r['validation'][population] for r in candidates]
                statistics[cohort][model][population]={k:{'mean':float(np.mean([m[k] for m in vals])),'sample_std':float(np.std([m[k] for m in vals],ddof=1))} for k in ['macro_auroc','macro_auprc','macro_precision','macro_recall','macro_f1']}
        full={r['seed']:r for r in rows if r['training_cohort']==cohort and r['model']=='unet1d'}
        enc={r['seed']:r for r in rows if r['training_cohort']==cohort and r['model']=='unet_encoder1d'}
        for population in COHORTS:
            differences=[full[s]['validation'][population]['macro_f1']-enc[s]['validation'][population]['macro_f1'] for s in SEEDS]
            paired[cohort][population]={'unet_minus_encoder_f1_by_seed':differences,'mean_difference':float(np.mean(differences)),'wins':sum(d>0 for d in differences)}
    return {'created_at':datetime.now().isoformat(),'protocol_sha256':sha(OUT/'protocol.json'),'runs':rows,'fixed_teacher_references':ref,
            'selected_native_validation':selected,'three_seed_statistics':statistics,'paired_decoder_ablation':paired,
            'fold10_evaluation':None,'new_teacher_or_deployment_automatically_replaced':False,'firmware_generated':False,
            'interpretation':['validation-only exploratory study, not new blind evaluation','select separately by training cohort/native validation; no mixed denominator ranking','reference InceptionTime weights trained on filtered cohort; unfiltered-training comparison is not matched for training population','encoder-only differs in decoder, pooled channels and parameter count','3 seeds do not establish significance or external generalization']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['all','prepare','train'],default='all');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available(),'This study uses the existing CUDA training environment'
    torch.set_num_threads(8)
    full=load_labeled_splits('D:/datasets/ptb-xl',100)
    guard=UNetReadGuard(pd.concat(list(full.values())));sys.addaudithook(guard.hook)
    frames={'unfiltered':{k:full[k] for k in ['train','validation']},'filtered':{k:full[k][full[k][C].sum(axis=1).gt(0)].copy() for k in ['train','validation']}}
    protocol=prepare(frames)
    if args.phase=='prepare':save(OUT/'read_guard_prepare.json',{'blocked_reads':guard.blocked,'reads':sorted(guard.reads)});return
    rows=train_runs(protocol,frames);ref=references(protocol,frames);summary=summarize(rows,ref)
    for path,expected in protocol['protected_artifacts'].items():assert sha(path)==expected,path
    summary['protected_artifacts_unchanged']=True
    save(OUT/'summary.json',summary)
    save(OUT/'read_guard.json',{'passed':not guard.blocked,'blocked_reads':guard.blocked,'reads':sorted(guard.reads),'fold10_evaluated':False})
    status('12 fresh runs and both validation cohorts completed; no fold10 evaluation')
    print(json.dumps({'selected':{cohort:{m:r['name'] for m,r in candidates.items()} for cohort,candidates in summary['selected_native_validation'].items()},'fold10_evaluation':None},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
