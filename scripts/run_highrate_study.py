"""PTB-XL 250/500 Hz raw-waveform study, two training cohorts, fold9 only.

No new 100-Hz training. Original 500-Hz files are SHA256 verified; 250 Hz
is obtained by a Kaiser FIR polyphase antialias filter. Original models,
shared sources, running U-Net study and firmware are never overwritten.
"""
from __future__ import annotations
import argparse, asyncio, copy, hashlib, json, os, re, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT=Path('D:/MyProject/college/ecg1')
sys.path.insert(0,str(ROOT))
from src.data.ptbxl import SUPERCLASSES, load_ecg
from src.training.data import load_labeled_splits, ensure_preprocessed_cache
from src.training.metrics import calculate_multilabel_metrics
from src.signal_processing.filters import FilterConfig
from src.models import build_model
from src.training.train import set_seed
from scripts.download_ptbxl import load_sha256_manifest, download_files

OUT=ROOT/'outputs/highrate_study_20261010'
SOURCE=Path('D:/datasets/ptb-xl')
RAW=OUT/'official_500hz'
CLI=Path('D:/STM32AI/User/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.1/Utilities/windows/stedgeai.exe')
MODELS=('cnn1d','inception1d','inception_time1d')
RATES=(250,500)
COHORTS=('filtered','unfiltered')
CLASSES=list(SUPERCLASSES)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.writing')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def status(message,**kwargs):
    print(message,flush=True);save(OUT/'status.json',dict(stage=message,time=datetime.now().isoformat(),**kwargs))

def frames():
    full=load_labeled_splits(SOURCE,100)
    assert len(full['train'])==17418 and len(full['validation'])==2183
    result={}
    for split in ('train','validation'):
        frame=full[split].reset_index(drop=True)
        assert frame.ecg_id.is_unique
        result[split]=frame
        for cohort in COHORTS:
            sub=frame if cohort=='unfiltered' else frame.loc[frame[CLASSES].sum(axis=1).gt(0)]
            result[cohort+'_'+split]=sub.reset_index(drop=True)
    assert len(result['filtered_train'])==17084 and len(result['filtered_validation'])==2146
    assert set(result['train'].patient_id).isdisjoint(result['validation'].patient_id)
    return result

def audit_resampler():
    t=np.arange(5000)/500
    low=np.sin(2*np.pi*20*t);high=np.sin(2*np.pi*180*t)
    a=resample_poly(low,1,2,window=('kaiser',8.6),padtype='line')
    b=resample_poly(high,1,2,window=('kaiser',8.6),padtype='line')
    low_rms=float(np.sqrt(np.mean(a[100:-100]**2)))
    alias_rms=float(np.sqrt(np.mean(b[100:-100]**2)))
    assert len(a)==2500 and abs(low_rms-2**-.5)<.01 and alias_rms<.002
    return dict(input_hz=500,output_hz=250,shape=[12,2500],pass_20hz_rms=low_rms,
                rejection_180hz_db=float(20*np.log10(max(alias_rms,1e-12)/(2**-.5))),
                naive_decimation_180hz_alias_hz=70,finite_record_padding='line',passed=True)

def freeze(f):
    OUT.mkdir(parents=True,exist_ok=True)
    protected={}
    for p in [ROOT/'outputs/teacher_preprocessing/selected_teacher.pt',
              ROOT/'outputs/distillation/distilled_seed44/checkpoint/cnn1d_best.pt',
              ROOT/'outputs/distillation/distilled_export/ptbxl_cnn1d_int8.onnx',
              ROOT/'outputs/validation_only_redo_20261010/single_teacher_kd_seed43/checkpoint/cnn1d_best.pt']:
        if p.exists():protected[str(p)]=sha(p)
    protocol=dict(date='2026-10-10',models=list(MODELS),rates=list(RATES),cohorts=list(COHORTS),
        seed=42,epochs=30,patience=8,batch_size=32,learning_rate=.001,weight_decay=.0001,
        new_100hz_training=False,source='PTB-XL v1.0.3 official records500',
        preprocessing={'500hz':'original mV waveform; no additional temporal filtering',
            '250hz':'500 Hz -> resample_poly(up=1,down=2,window=(kaiser,8.6),padtype=line)',
            'normalization':'per-lead global mean/std fitted only on each training cohort',
            'bandpass':None,'baseline_removal':False,'notch':None},
        checkpoint_selection='fold9 native cohort macro AUROC; minimum improvement .0001',
        final_ranking='native fold9 macro F1, fine thresholds chosen there only',
        cross_cohort_thresholds='one threshold vector fitted on training-native fold9 cohort; reused on both',
        threshold_grid='0.01..0.99 step0.01, closest to .5 on ties',
        model_kernels='existing sample-domain kernels/strides unchanged; physical receptive fields shrink with fs',
        baseline_comparability='existing 100 Hz models may differ in preprocessing/cohort; report explicitly',
        test_fold10_used=False,multiseed_confirmed=False,protected_artifacts=protected,
        counts={k:len(v) for k,v in f.items()},resampling_audit=audit_resampler())
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    for k,v in f.items():v[['ecg_id','patient_id','strat_fold',*CLASSES,'filename_hr','filename_lr']].to_csv(OUT/f'{k}_cohort.csv',index=False)
    status('Original protocol preserved; revised scope: six250-Hz and four500-Hz runs; no100-Hz retraining')
    return protocol

def download(f,workers=64,limit=None):
    RAW.mkdir(exist_ok=True)
    for name in ['ptbxl_database.csv','scp_statements.csv','SHA256SUMS.txt','LICENSE.txt']:
        target=RAW/name
        if target.exists():assert sha(target)==sha(SOURCE/name)
        else:shutil.copy2(SOURCE/name,target)
    manifest=load_sha256_manifest(RAW/'SHA256SUMS.txt')
    names=pd.concat([f['train'],f['validation']]).filename_hr.tolist()
    selected=[]
    for name in names:
        for suffix in ('.hea','.dat'):
            relative=name+suffix
            assert relative in manifest and relative.startswith('records500/') and '..' not in Path(relative).parts
            selected.append((relative,manifest[relative]))
    if limit:selected=selected[:limit]
    status(f'Downloading/verifying {len(selected)} official 500-Hz files (folds1-9 only)')
    moved,count,errors=asyncio.run(download_files(RAW,selected,workers))
    save(OUT/'download_audit.json',dict(records=len(names),files_requested=len(selected),files_verified=count,
         transferred=moved,failures=errors,sha256_manifest=sha(RAW/'SHA256SUMS.txt'),
         source='https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/',fold10_downloaded=False))
    if errors or count!=len(selected):raise RuntimeError(f'Download incomplete: {errors[:3]}')

def prepare_cache(f):
    cache=OUT/'cache';cache.mkdir(exist_ok=True)
    for split in ('train','validation'):
        frame=f[split]
        paths={fs:cache/f'raw_{split}_{fs}hz.npy' for fs in RATES}
        meta=cache/f'raw_{split}.json'
        fingerprint=hashlib.sha256(json.dumps(dict(ids=frame.ecg_id.tolist(),preprocessing=read(OUT/'protocol.json')['preprocessing'] if False else read(OUT/'protocol.json')['preprocessing']),sort_keys=True).encode()).hexdigest()
        if meta.exists():
            assert read(meta)['fingerprint']==fingerprint
            for fs,path in paths.items():assert np.load(path,mmap_mode='r').shape==(len(frame),12,fs*10)
            continue
        arrays={fs:np.lib.format.open_memmap(p.with_suffix('.building'),mode='w+',dtype=np.float32,shape=(len(frame),12,fs*10)) for fs,p in paths.items()}
        def one(name):
            rec=load_ecg(RAW,name,sampling_rate_hz=500)
            x=np.asarray(rec.signal,dtype=np.float32)
            assert x.shape==(5000,12) and np.isfinite(x).all()
            return x.T.copy(),resample_poly(x,1,2,axis=0,window=('kaiser',8.6),padtype='line').T.astype(np.float32)
        started=time.time()
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i,(x500,x250) in enumerate(pool.map(one,frame.filename_hr)):
                arrays[500][i]=x500;arrays[250][i]=x250
                if (i+1)%1000==0 or i+1==len(frame):status(f'Cache {split}: {i+1}/{len(frame)}, {time.time()-started:.0f}s')
        for fs,a in arrays.items():a.flush()
        arrays.clear();del a
        for fs,p in paths.items():p.with_suffix('.building').replace(p)
        save(meta,dict(fingerprint=fingerprint,shape_by_fs={fs:[len(frame),12,fs*10] for fs in RATES},ecg_ids=frame.ecg_id.tolist(),lead_names=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']))
    status('Raw 500 Hz and antialiased 250 Hz caches complete')

def moments(array,positions):
    total=np.zeros(12,dtype=np.float64);squares=np.zeros(12,dtype=np.float64);n=0
    for start in range(0,len(positions),64):
        x=np.asarray(array[positions[start:start+64]],dtype=np.float64)
        total+=x.sum(axis=(0,2));squares+=(x*x).sum(axis=(0,2));n+=len(x)*x.shape[-1]
    mean=total/n;std=np.maximum(np.sqrt(np.maximum(squares/n-mean*mean,0)),1e-8)
    return mean,std

class Waves(Dataset):
    def __init__(self,array,frame,positions,mean,std):
        self.array=array;self.y=frame[CLASSES].to_numpy(dtype=np.float32);self.positions=np.asarray(positions)
        self.mean=np.asarray(mean,dtype=np.float32)[:,None];self.std=np.asarray(std,dtype=np.float32)[:,None]
    def __len__(self):return len(self.positions)
    def __getitem__(self,i):
        x=(self.array[self.positions[i]]-self.mean)/self.std
        return torch.from_numpy(np.ascontiguousarray(x)),torch.from_numpy(self.y[i].copy())

def fine(y,p):
    p=np.asarray(p,dtype=np.float64);grid=np.arange(1,100,dtype=np.float64)/100
    ans=[]
    for i in range(5):
        truth=y[:,i].astype(bool);pred=p[:,i,None]>=grid[None,:]
        tp=(pred&truth[:,None]).sum(axis=0);fp=(pred&~truth[:,None]).sum(axis=0);fn=(~pred&truth[:,None]).sum(axis=0)
        scores=2*tp/np.maximum(2*tp+fp+fn,1)
        tied=grid[np.isclose(scores,scores.max())];ans.append(float(tied[np.argmin(abs(tied-.5))]))
    return ans
def metric(y,p,thresholds):
    m=calculate_multilabel_metrics(y,p,thresholds)
    aps=average_precision_score(y,p,average=None);m['macro_auprc']=float(aps.mean())
    for i,c in enumerate(CLASSES):m['per_class'][c]['auprc']=float(aps[i])
    assert abs(m['macro_f1']-f1_score(y,np.asarray(p,dtype=np.float64)>=thresholds,average='macro',zero_division=0))<1e-10
    return m

def epoch(model,loader,criterion,device,optimizer=None,scaler=None,amp=True):
    train=optimizer is not None;model.train(train);loss_sum=0;n=0;ys=[];ps=[]
    for x,y in loader:
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        if train:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train),torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
            logits=model(x);loss=criterion(logits,y)
        if train:
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        loss_sum+=float(loss.detach())*len(x);n+=len(x)
        if not train:ys.append(y.cpu().numpy());ps.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
    return loss_sum/n,None if train else np.concatenate(ys),None if train else np.concatenate(ps)

def run_one(f,model_name,fs,cohort,batch):
    folder=OUT/cohort/f'{model_name}_{fs}hz_seed42';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return read(folder/'result.json')
    train=f[cohort+'_train'];native=f[cohort+'_validation'];allval=f['validation']
    trainpos=pd.Index(f['train'].ecg_id).get_indexer(train.ecg_id)
    nativepos=pd.Index(allval.ecg_id).get_indexer(native.ecg_id)
    trainarray=np.load(OUT/f'cache/raw_train_{fs}hz.npy',mmap_mode='r')
    valarray=np.load(OUT/f'cache/raw_validation_{fs}hz.npy',mmap_mode='r')
    statpath=OUT/f'cache/{cohort}_{fs}hz_statistics.json'
    if statpath.exists():s=read(statpath);mean=np.array(s['mean']);std=np.array(s['std'])
    else:
        mean,std=moments(trainarray,trainpos);save(statpath,dict(mean=mean.tolist(),std=std.tolist(),fitted_ecg_ids=train.ecg_id.tolist(),folds=list(range(1,9))))
    set_seed(42);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator=torch.Generator().manual_seed(42)
    tdata=Waves(trainarray,train,trainpos,mean,std);vdata=Waves(valarray,native,nativepos,mean,std)
    tl=DataLoader(tdata,batch_size=batch,shuffle=True,generator=generator,num_workers=0,pin_memory=device.type=='cuda')
    vl=DataLoader(vdata,batch_size=batch,num_workers=0,pin_memory=device.type=='cuda')
    model=build_model(model_name,dropout=.2).to(device)
    positives=train[CLASSES].sum().to_numpy();pw=(len(train)-positives)/np.maximum(positives,1)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,dtype=torch.float32,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    config=dict(dataset=dict(root=str(RAW),sampling_rate_hz=fs,require_diagnostic_label=cohort=='filtered',train_folds=list(range(1,9)),validation_fold=9,test_fold=10),
        signal_processing=dict(normalization='train_global',training_mean=mean.tolist(),training_std=std.tolist(),baseline_removal=False,bandpass=None,notch_hz=None),
        model=dict(type=model_name,dropout=.2),training=dict(epochs=30,early_stopping_patience=8,batch_size=batch,random_seed=42,learning_rate=.001,weight_decay=.0001,use_amp=True),
        highrate_preprocessing=read(OUT/'protocol.json')['preprocessing'])
    save(folder/'config.json',config)
    history=[];best=-np.inf;stale=0;best_epoch=0;started=time.monotonic();checkpoint=folder/f'{model_name}_best.pt'
    if (folder/'history.json').exists():raise RuntimeError(f'Partial run requires inspection before retraining: {folder}')
    status(f'Training {cohort}/{model_name}/{fs}Hz (batch{batch}, {len(train)} records)')
    for ep in range(1,31):
        t=time.monotonic()
        loss,_,_=epoch(model,tl,criterion,device,optimizer,scaler)
        vloss,y,p=epoch(model,vl,criterion,device)
        m=metric(y,p,[.5]*5);score=m['macro_auroc']
        history.append(dict(epoch=ep,train_loss=loss,validation_loss=vloss,validation_macro_auroc=score,validation_macro_f1_at_05=m['macro_f1'],seconds=time.monotonic()-t))
        if score>best+1e-4:
            best=score;stale=0;best_epoch=ep
            torch.save(dict(model_type=model_name,model_state=model.state_dict(),epoch=ep,scope='full',training_records=len(train),validation_records=len(native),sampling_rate_hz=fs,
                superclasses=CLASSES,lead_names=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'],thresholds=[.5]*5,pos_weight=pw.tolist(),config=config,
                input_contract='use this highrate runner; legacy preprocessing/export scripts do not support this raw/250-Hz contract'),checkpoint)
        else:stale+=1
        save(folder/'history.json',dict(best_epoch=best_epoch,history=history,completed=False))
        status(f'{cohort}/{model_name}/{fs}Hz epoch{ep}: AUROC={score:.6f}, F1(.5)={m["macro_f1"]:.6f}, {history[-1]["seconds"]:.1f}s')
        if stale>=8:break
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(cp['model_state']);model.to(device)
    full_data=Waves(valarray,allval,np.arange(len(allval)),mean,std)
    _,y,p=epoch(model,DataLoader(full_data,batch_size=batch,num_workers=0),criterion,device,amp=False)
    thresholds=fine(y[nativepos],p[nativepos]);cohort_metrics={}
    for evaluated in COHORTS:
        positions=pd.Index(allval.ecg_id).get_indexer(f[evaluated+'_validation'].ecg_id)
        cohort_metrics[evaluated]=metric(y[positions],p[positions],thresholds)
        cohort_metrics[evaluated+'_05']=metric(y[positions],p[positions],[.5]*5)
    np.savez_compressed(folder/'validation_predictions.npz',ecg_ids=allval.ecg_id.to_numpy(),patient_ids=allval.patient_id.to_numpy(),targets=y,probabilities=p,thresholds=thresholds)
    cp.update(thresholds=thresholds,validation_metrics_tuned=cohort_metrics[cohort]);torch.save(cp,checkpoint)
    result=dict(model=model_name,fs=fs,training_cohort=cohort,seed=42,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),best_epoch=best_epoch,
        epochs_completed=len(history),batch_size=batch,training_records=len(train),parameters=sum(v.numel() for v in model.parameters()),
        thresholds=thresholds,validation=cohort_metrics,elapsed_seconds=time.monotonic()-started,fold10_used=False)
    save(folder/'history.json',dict(best_epoch=best_epoch,history=history,completed=True));save(folder/'result.json',result)
    del model,optimizer,trainarray,valarray;torch.cuda.empty_cache()
    status(f'Complete {cohort}/{model_name}/{fs}Hz native F1={cohort_metrics[cohort]["macro_f1"]:.6f}')
    return result

def baseline_references(f):
    refs=[('inception_time1d','filtered',ROOT/'outputs/teacher_preprocessing/inception_time1d_P0_current_seed42/checkpoint/inception_time1d_best.pt'),
          ('inception1d','filtered',ROOT/'outputs/teacher_preprocessing/inception1d_P0_current_seed42/checkpoint/inception1d_best.pt'),
          ('cnn1d','filtered',ROOT/'outputs/validation_only_redo_20261010/supervised_seed42/checkpoint/cnn1d_best.pt'),
          ('inception1d','unfiltered',ROOT/'outputs/optimization/08_global_inception1d/checkpoint/inception1d_best.pt'),
          ('cnn1d','unfiltered',ROOT/'outputs/checkpoints/baseline_best.pt')]
    results=[]
    for name,label,path in refs:
        if not path.exists():continue
        dest=OUT/'references_100hz'/f'{name}_{label}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():results.append(read(dest/'result.json'));continue
        status(f'Evaluating existing 100 Hz reference {name}/{label}; no retraining')
        cp=torch.load(path,map_location='cpu',weights_only=True)
        actual=cp.get('model_type','cnn1d');assert actual==name
        settings=cp['config']['signal_processing']
        cache=ensure_preprocessed_cache(f['validation'],SOURCE,OUT/'reference_cache',split_name='validation_reference',sampling_rate_hz=100,config=FilterConfig.from_mapping(settings))
        a=np.load(cache,mmap_mode='r');mean=np.zeros(12);std=np.ones(12)
        data=Waves(a,f['validation'],np.arange(len(a)),mean,std)
        model=build_model(name,dropout=cp['config']['model']['dropout']).cuda().eval();model.load_state_dict(cp['model_state'])
        native_filtered=cp['config']['dataset'].get('require_diagnostic_label',False)
        native='filtered' if native_filtered else 'unfiltered'
        assert native==label,(path,native,label)
        criterion=torch.nn.BCEWithLogitsLoss()
        _,y,p=epoch(model,DataLoader(data,batch_size=32),criterion,torch.device('cuda'),amp=False)
        nativepos=pd.Index(f['validation'].ecg_id).get_indexer(f[native+'_validation'].ecg_id)
        thresholds=fine(y[nativepos],p[nativepos]);metrics={}
        for evaluated in COHORTS:
            pos=pd.Index(f['validation'].ecg_id).get_indexer(f[evaluated+'_validation'].ecg_id)
            metrics[evaluated]=metric(y[pos],p[pos],thresholds);metrics[evaluated+'_05']=metric(y[pos],p[pos],[.5]*5)
        np.savez_compressed(dest/'validation_predictions.npz',ecg_ids=f['validation'].ecg_id.to_numpy(),patient_ids=f['validation'].patient_id.to_numpy(),targets=y,probabilities=p,thresholds=thresholds)
        r=dict(model=name,fs=100,training_cohort=native,checkpoint=str(path),checkpoint_sha256=sha(path),
            seed=cp['config']['training'].get('random_seed'),best_epoch=cp['epoch'],training_records=cp['training_records'],
            preprocessing=settings,training_config=cp['config']['training'],validation=metrics,thresholds=thresholds,
            weights_reused=True,fold10_used=False,comparison_limitation='Existing 100 Hz filtering/normalization differs from raw highrate pipeline; single-seed exploratory system comparison')
        save(dest/'result.json',r);results.append(r);del model;torch.cuda.empty_cache()
    return results

def resource(row,f):
    """Actual ONNX export, training-only INT8 and Cube.AI, not board validation."""
    import onnx, onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, quantize_static, CalibrationMethod, QuantFormat, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
    folder=OUT/'resources'/f'{row["training_cohort"]}_{row["model"]}_{row["fs"]}hz';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'resource.json').exists():return read(folder/'resource.json')
    status('Export/quantize/Cube.AI '+folder.name)
    cp=torch.load(row['checkpoint'],map_location='cpu',weights_only=True)
    model=build_model(row['model'],dropout=.2).cpu().eval();model.load_state_dict(cp['model_state'])
    sample=torch.zeros(1,12,row['fs']*10);fp=folder/'model_float32.onnx';qp=folder/'model_int8.onnx'
    previous_hashes={version:sha(path) if path.exists() else None for version,path in [('float32',fp),('int8',qp)]}
    torch.onnx.export(model,sample,str(fp),input_names=['ecg'],output_names=['logits'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp)))
    train=f[row['training_cohort']+'_train'];pos=pd.Index(f['train'].ecg_id).get_indexer(train.ecg_id)
    array=np.load(OUT/f'cache/raw_train_{row["fs"]}hz.npy',mmap_mode='r')
    mean=np.array(cp['config']['signal_processing']['training_mean'],dtype=np.float32)[None,:,None]
    std=np.array(cp['config']['signal_processing']['training_std'],dtype=np.float32)[None,:,None]
    chosen=np.random.default_rng(42).choice(len(train),size=128 if row['model']=='cnn1d' else 32,replace=False)
    calib=np.ascontiguousarray((array[pos[chosen]]-mean)/std)
    np.savez_compressed(folder/'calibration_train.npz',signals=calib,ecg_ids=train.iloc[chosen].ecg_id.to_numpy(),folds=train.iloc[chosen].strat_fold.to_numpy())
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(fp),opts,providers=['CPUExecutionProvider'])
    with torch.inference_mode():expected=model(torch.from_numpy(calib[:8])).numpy()
    actual=np.concatenate([session.run(None,{'ecg':x[None]})[0] for x in calib[:8]])
    error=float(abs(expected-actual).max());assert error<1e-3,error
    # Native ONNX shape inference on Windows cannot open a Unicode temp path.
    import tempfile
    temporary=folder/'quantization_tmp';temporary.mkdir(exist_ok=True)
    previous_temp=tempfile.tempdir;tempfile.tempdir=str(temporary)
    pre=folder/'preprocessed_float32.onnx'
    try:quant_pre_process(str(fp),str(pre),skip_symbolic_shape=True)
    finally:tempfile.tempdir=previous_temp
    class Reader(CalibrationDataReader):
        def __init__(self):self.it=iter(calib)
        def get_next(self):
            v=next(self.it,None);return None if v is None else {'ecg':v[None]}
    quantize_static(str(pre),str(qp),Reader(),quant_format=QuantFormat.QDQ,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,
        per_channel=True,calibrate_method=CalibrationMethod.MinMax,extra_options={'WeightSymmetric':True,'ActivationSymmetric':False},calibration_providers=['CPUExecutionProvider'])
    onnx.checker.check_model(onnx.load(str(qp)))
    resources={}
    for version,path in [('float32',fp),('int8',qp)]:
        dest=folder/version;dest.mkdir(exist_ok=True);name=f'ecg_{row["model"]}_{row["fs"]}_{version}'
        cmd=[str(CLI),'analyze','--model',str(path),'--name',name,'--target','stm32f4','--compression','none','--optimization','ram','--output',str(dest),
            '--workspace',str(folder/(version+'_workspace')),'--inputs-ch-position','chfirst','--output-data-type','float32','--c-api','legacy']
        cached_report=dest/f'{name}_analyze_report.txt';cached_info=dest/f'{name}_c_info.json'
        returncode=0
        if not (previous_hashes[version]==sha(path) and cached_report.exists() and cached_info.exists()):
            with (folder/(version+'_cubeai.log')).open('w',encoding='utf-8') as log:
                returncode=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
        if returncode:
            resources[version]=dict(conversion_success=False,error_log=str(folder/(version+'_cubeai.log')));continue
        info=read(dest/f'{name}_c_info.json');txt=(dest/f'{name}_analyze_report.txt').read_text(encoding='utf-8')
        mem=info['memory_footprint'];macc=int(re.search(r'^macc\s+:\s+([\d,]+)',txt,re.M).group(1).replace(',',''))
        weights=int(mem['weights']);act=int(mem['activations'])
        # Reference overhead is an estimate; activation includes I/O in this project export.
        flash=weights+(158876-112276);ram=act+(43672-24000)
        resources[version]=dict(conversion_success=True,weights_bytes=weights,activations_bytes=act,macc=macc,memory_footprint=mem,
            estimated_complete_flash=flash,estimated_complete_main_sram=ram,overhead_source='old CNN seed44 link map; estimates only, no new firmware',
            fits_f407ve_flash_estimate=flash<=512*1024,fits_f407vg_flash_estimate=flash<=1024*1024,fits_main_sram_estimate=ram<=128*1024)
    quant_metrics=None
    if row['model']=='cnn1d':
        qs=ort.InferenceSession(str(qp),opts,providers=['CPUExecutionProvider'])
        valarray=np.load(OUT/f'cache/raw_validation_{row["fs"]}hz.npy',mmap_mode='r');out=[]
        for i,x in enumerate(valarray):
            logits=qs.run(None,{'ecg':np.ascontiguousarray((x[None]-mean)/std)})[0]
            out.append(1/(1+np.exp(-np.clip(logits,-80,80))))
        p=np.concatenate(out);y=f['validation'][CLASSES].to_numpy(dtype=np.uint8);quant_metrics={}
        for cohort in COHORTS:
            ix=pd.Index(f['validation'].ecg_id).get_indexer(f[cohort+'_validation'].ecg_id)
            quant_metrics[cohort]=metric(y[ix],p[ix],row['thresholds'])
        np.savez_compressed(folder/'int8_validation_predictions.npz',ecg_ids=f['validation'].ecg_id.to_numpy(),targets=y,probabilities=p,thresholds=row['thresholds'])
    result=dict(model=row['model'],fs=row['fs'],training_cohort=row['training_cohort'],checkpoint_sha256=row['checkpoint_sha256'],onnx_max_logits_error=error,
        calibration_records=len(calib),calibration_folds=sorted({int(v) for v in train.iloc[chosen].strat_fold}),resources=resources,int8_validation=quant_metrics,
        int8_thresholds_retuned=False,new_firmware_compiled=False,physical_board_validated=False,
        serial_fp32_115200_min_seconds=12*row['fs']*10*4*10/115200,serial_int8_115200_min_seconds=12*row['fs']*10*10/115200)
    save(folder/'resource.json',result);return result

def independent_audit(f,rows,refs,resources):
    report=dict(passed=True,new_training_runs=len(rows),new_training_rates=sorted({r['fs'] for r in rows}),new_100hz_training=0,
        fold10_inference=0,patient_isolation=True,counts={k:len(v) for k,v in f.items()},cohort_metrics_checked=0,
        resampling=audit_resampler(),protected_artifacts_unchanged=True,board_validation=False)
    assert len(rows)==10 and {r['training_cohort'] for r in rows}==set(COHORTS)
    expected={(m,h,c) for m in MODELS for c in COHORTS for h in RATES if not (m=='inception_time1d' and h==500)}
    assert {(r['model'],r['fs'],r['training_cohort']) for r in rows}==expected
    for r in rows+refs:
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=True)
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        folder=Path(r['checkpoint']).parent if r['fs']!=100 else OUT/'references_100hz'/f'{r["model"]}_{r["training_cohort"]}'
        a=np.load(folder/'validation_predictions.npz');np.testing.assert_array_equal(a['ecg_ids'],f['validation'].ecg_id)
        np.testing.assert_array_equal(a['targets'],f['validation'][CLASSES])
        native=pd.Index(f['validation'].ecg_id).get_indexer(f[r['training_cohort']+'_validation'].ecg_id)
        np.testing.assert_allclose(fine(a['targets'][native],a['probabilities'][native]),r['thresholds'],atol=0)
        for cohort in COHORTS:
            ix=pd.Index(f['validation'].ecg_id).get_indexer(f[cohort+'_validation'].ecg_id)
            m=metric(a['targets'][ix],a['probabilities'][ix],r['thresholds'])
            for key in ('macro_f1','macro_precision','macro_recall','macro_auroc','macro_auprc'):assert abs(m[key]-r['validation'][cohort][key])<1e-10
            for j,c in enumerate(CLASSES):
                pred=np.asarray(a['probabilities'][ix,j],dtype=np.float64)>=r['thresholds'][j];truth=a['targets'][ix,j].astype(bool)
                tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum());tn=len(ix)-tp-fp-fn
                assert m['per_class'][c]['confusion_matrix']==[[tn,fp],[fn,tp]]
            report['cohort_metrics_checked']+=1
    for p,digest in read(OUT/'protocol.json')['protected_artifacts'].items():assert sha(p)==digest
    save(OUT/'audit.json',report);return report

def update_docs(summary):
    """Read current shared docs at the last moment, append a uniquely marked block."""
    rows=summary['runs'];refs=summary['references_100hz'];res=summary['resources']
    intro=['### 250/500 Hz原始波形探索（2026-10-10）','',
      '按用户收缩范围，仅新增10次seed42训练：250 Hz保留轻量CNN、紧凑Inception、InceptionTime，筛选/未筛选各训练；500 Hz仅保留已完成的CNN与紧凑Inception，两队列共4次，不再训练500 Hz InceptionTime。100 Hz直接复用现有检查点，没有重训。',
      '训练折1–8、验证折9；只对对应训练队列拟合全局导联标准化。500 Hz读取官方原始mV波形，不另做中值去基线或40 Hz低通；250 Hz使用Kaiser FIR多相抗混叠降采样。',
      '**可比性：**100 Hz旧模型大多有0.5–40 Hz滤波/去基线，高采样率本轮用原始波形；因此这是输入方案的探索性系统比较，不能把全部差异归因于采样率。采样点定义的卷积核/步长固定，物理感受野随采样率缩短。',
      '所有新模型从随机初始化训练，最多30轮、8轮早停、AdamW/BCE，单种子不代表稳定提升。原蒸馏CNN只作为部署历史参考；本轮CNN为监督训练，没有跨采样率蒸馏。',
      '每个训练版本在自己对应的折9队列细搜阈值（0.01步长），同一组阈值复用到筛选与未筛选验证队列。未筛选保留五类全零标签，它们不等于正常。折10本轮未推理。',
      '证据与复现：`outputs/highrate_study_20261010/`；`scripts/run_highrate_study.py --phase all`。','']
    table=['|训练队列|网络|Hz|最佳轮|验证队列|Precision|Recall|F1|AUROC|HYP Recall|','|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in refs+rows:
        for cohort in COHORTS:
            m=r['validation'][cohort]
            table.append(f'|{r["training_cohort"]}|{r["model"]}|{r["fs"]}|{r["best_epoch"]}|{cohort}|{m["macro_precision"]:.6f}|{m["macro_recall"]:.6f}|{m["macro_f1"]:.6f}|{m["macro_auroc"]:.6f}|{m["per_class"]["HYP"]["recall"]:.6f}|')
    conclusions=[]
    for r in rows:
        candidates=[v for v in refs if v['model']==r['model'] and v['training_cohort']==r['training_cohort']]
        if candidates:
            base=candidates[0];cohort=r['training_cohort'];delta=(r['validation'][cohort]['macro_f1']-base['validation'][cohort]['macro_f1'])*100
            conclusions.append(f'- {cohort} / {r["model"]} / {r["fs"]} Hz：相对已有同训练队列100 Hz，折9宏F1差值{delta:+.3f}个百分点；预处理差异限制见上。')
        else:conclusions.append(f'- {r["training_cohort"]} / {r["model"]} / {r["fs"]} Hz：无匹配的已有100 Hz训练队列检查点，不能给出严格同队列增益。')
    detail=['','#### 整体与逐类验证指标','']+table+['','|训练队列|网络|Hz|评估队列|类别|AUROC|Precision|Recall|F1|AUPRC|阈值|','|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in refs+rows:
        for cohort in COHORTS:
            for c in CLASSES:
                m=r['validation'][cohort]['per_class'][c]
                detail.append(f'|{r["training_cohort"]}|{r["model"]}|{r["fs"]}|{cohort}|{c}|'+ '|'.join(f'{m[k]:.6f}' for k in ['auroc','precision','recall','f1','auprc','threshold'])+'|')
    deploy=['### 250/500 Hz新模型资源分析（2026-10-10）','',
        'STM32F407VG：1 MiB Flash、128 KiB主SRAM+64 KiB CCM；旧VE工程Flash为512 KiB。CCM独立寻址，不能把192 KiB当成单一主SRAM区，DMA亦不能直接访问CCM。',
        '以下权重、激活、MACC来自本轮实际Cube.AI分析；完整固件估算暂借旧学生额外Flash 46600 B、额外主SRAM 19672 B，未做新固件链接，不代表板端验证。模型激活中的输入/输出不能再次重复相加。',
        '输入12导联×10秒，250/500 Hz的INT8输入分别30000/60000 B，FP32分别120000/240000 B。现有协议电脑预处理后分块发送FP32并在板端量化；板上不需要保留整片FP32缓存。','',
        '|网络|Hz|训练队列|版本|权重B|激活B|MACC|估算固件FlashB|估算主SRAM B|主SRAM可容纳|','|---|---:|---|---|---:|---:|---:|---:|---:|---|']
    for r in res:
        for ver,m in r['resources'].items():
            if m['conversion_success']:deploy.append(f'|{r["model"]}|{r["fs"]}|{r["training_cohort"]}|{ver}|{m["weights_bytes"]}|{m["activations_bytes"]}|{m["macc"]}|{m["estimated_complete_flash"]}|{m["estimated_complete_main_sram"]}|{m["fits_main_sram_estimate"]}|')
            else:deploy.append(f'|{r["model"]}|{r["fs"]}|{r["training_cohort"]}|{ver}|转换失败|—|—|—|—|查原日志|')
    deploy+=['','115200/8N1只计数据的FP32发送下限：100/250/500 Hz为4.17/10.42/20.83秒，实际ASCII浮点协议、ACK与CRC更慢。改为二进制INT8传输后的下限为1.04/2.60/5.21秒，尚未实现该协议。MACC不能直接换算真实推理时延。',
        '当前HEX、阈值、UART固定样本数仍是100 Hz；新模型要改协议输入长度、量化参数、模型C代码、工程目标与完整链接，再做真实烧录/一致性/耗时/栈峰值验证。','']
    blocks={'md/模型训练与优化.md':'\n'.join(intro+conclusions+['']+table),
        'md/各模型整体与逐类指标总表.md':'\n'.join(intro+detail),
        'md/STM32部署与串口验证.md':'\n'.join(deploy),
        'md/README.md':'### 250/500 Hz探索完成\n\n新增10次训练（250 Hz三网络×两队列；500 Hz仅CNN与紧凑Inception×两队列），100 Hz只复用已有模型。结果见[训练记录](模型训练与优化.md#highrate-study-20261010)、[指标表](各模型整体与逐类指标总表.md#highrate-study-20261010)、[资源](STM32部署与串口验证.md#highrate-study-20261010)。新模型未替换固件；原始高采样率与旧100 Hz滤波条件不同，结果按单种子系统比较解释。'}
    for relative,body in blocks.items():
        path=ROOT/relative;original=path.read_text(encoding='utf-8');digest=sha(path)
        backup=OUT/'documentation_backups'/path.name
        if not backup.exists():backup.parent.mkdir(exist_ok=True);shutil.copy2(path,backup)
        start='<!-- highrate-study-20261010:start -->';end='<!-- highrate-study-20261010:end -->'
        payload=start+'\n<a id="highrate-study-20261010"></a>\n\n'+body.strip()+'\n'+end
        if start in original:updated=re.sub(re.escape(start)+r'.*?'+re.escape(end),lambda _:payload,original,flags=re.S)
        else:updated=original.rstrip()+'\n\n'+payload+'\n'
        assert sha(path)==digest,'Concurrent documentation modification; rerun report'
        temp=path.with_suffix('.highrate-writing');temp.write_text(updated,encoding='utf-8');temp.replace(path)
    save(OUT/'comparison.json',dict(conclusions=conclusions,baseline_limits='different preprocessing; unfiltered InceptionTime100 missing',new_100hz_training=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['probe','prepare','train','resources','report','all'],default='all');parser.add_argument('--batch-size',type=int,default=32);parser.add_argument('--workers',type=int,default=64)
    args=parser.parse_args();f=frames();protocol=freeze(f)
    if args.phase=='probe':download(f,args.workers,limit=8);return
    if args.phase in ('prepare','all'):
        download(f,args.workers);prepare_cache(f)
        if args.phase=='prepare':return
    rows=[]
    for model_name in MODELS:
        for cohort in COHORTS:
            for fs in RATES:
                if model_name=='inception_time1d' and fs==500:continue
                if args.phase in ('train','all'):rows.append(run_one(f,model_name,fs,cohort,args.batch_size))
                else:rows.append(read(OUT/cohort/f'{model_name}_{fs}hz_seed42/result.json'))
    if args.phase=='train':return
    refs=baseline_references(f);resources=[]
    for r in rows:
        # Both CNN cohorts measured; teacher graph sizes depend on architecture/fs,
        # so filtered teacher exports suffice for resource screening.
        if r['model']=='cnn1d' or r['training_cohort']=='filtered':resources.append(resource(r,f))
    if args.phase=='resources':return
    audit=independent_audit(f,rows,refs,resources)
    summary=dict(protocol=protocol,scope_revision=read(OUT/'scope_revision.json'),runs=rows,references_100hz=refs,resources=resources,audit=audit,completed_at=datetime.now().isoformat(),fold10_used=False)
    save(OUT/'summary.json',summary);update_docs(summary)
    status('Complete: 10 new runs (six250, four500), old100 comparison, both cohorts, actual Cube.AI, audited documents')

if __name__=='__main__':main()
