"""Prespecified B experiment: raw ECG noise, paired training, distillation, INT8."""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest, get_split
from src.data.ptbxl import LEAD_NAMES, SUPERCLASSES, load_ecg
from src.models import build_model
from src.signal_processing.filters import FilterConfig, preprocess_ecg
from src.training.noise import add_raw_gaussian_noise, snr_db, NoiseViewsDataset
from src.training.distillation import multilabel_distillation_loss
from src.training.metrics import calculate_multilabel_metrics, optimize_thresholds
from src.training.train import set_seed, _run_epoch, _save_checkpoint
from src.training.data import load_labeled_splits, fit_training_normalization, ensure_preprocessed_cache

OUT = ROOT / 'outputs/noise_robustness_b'
BASE = ROOT / 'outputs/distillation/distilled_seed44/checkpoint/cnn1d_best.pt'
OLD_INT8 = ROOT / 'outputs/distillation/distilled_export/ptbxl_cnn1d_int8.onnx'
CLI = Path('D:/STM32AI/User/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.1/Utilities/windows/stedgeai.exe')
NOISE_SEED = 20261003
SEEDS = [42, 43, 44]
LEVELS = {'train':[20,10], 'validation':[20,10], 'test':[30,20,10]}
TZ = datetime.timezone(datetime.timedelta(hours=8))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.saving')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def status(stage, **extra):
    print(stage, flush=True)
    save(OUT/'status.json',{'stage':stage,'updated_at':datetime.datetime.now(TZ).isoformat(),**extra})


def load(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def initialize():
    base=load(BASE)
    config=copy.deepcopy(base['config'])
    assert base['training_records']==17084 and base['validation_records']==2146
    assert config['dataset']['require_diagnostic_label'] is True
    frames=load_labeled_splits(config['dataset']['root'],100,require_diagnostic_label=True)
    expected={'train':17084,'validation':2146,'test':2158}
    assert {name:len(frame) for name,frame in frames.items()}==expected
    patients={name:set(frame.patient_id) for name,frame in frames.items()}
    assert not (patients['train']&patients['validation'] or patients['train']&patients['test'] or patients['validation']&patients['test'])
    protected=[BASE, OLD_INT8, ROOT/'outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_STM32F407VE.hex']
    protected += [ROOT/f'outputs/distillation/{arm}_seed{seed}/checkpoint/cnn1d_best.pt' for arm in ['baseline','distilled'] for seed in SEEDS]
    protocol={
        'experiment':'B_raw_gaussian_noise','date':'2026-10-03','noise_seed':NOISE_SEED,
        'noise_type':'Gaussian white noise only; no synthetic ECG, no EMG or electrode-motion simulation',
        'noise_stage':'raw mV before baseline removal, bandpass and training-global normalization',
        'snr_definition':'per-lead demeaned signal power / added noise power, before preprocessing',
        'train_view_probabilities':[.5,.25,.25],'train_views':['clean','20dB','10dB'],
        'fixed_noisy_views_per_record':2,'noise_levels':LEVELS,
        'cohort':expected,'patient_isolation_verified':True,
        'teacher':{'type':'inception1d','seeds':[44],'arms':['clean','noise'],'initialization':'from scratch'},
        'student':{'type':'cnn1d','seeds':SEEDS,'arms':['noise_supervised','noise_distilled']},
        'training':{'epochs':30,'early_stopping_patience':8,'batch_size':32,'lr':.001,'weight_decay':.0001,
                    'checkpoint_selection':'clean validation AUROC, improvement > 0.0001',
                    'threshold_selection':'clean validation only, unchanged coarse grid'},
        'distillation':{'alpha':.5,'temperature':2,'teacher_input':'same selected record and noise view as student',
                        'teacher_logits':'new view-aligned cache, never legacy clean-only cache'},
        'deployment_selection':{'candidates':'three noise-distilled students',
             'score':'0.50 clean validation F1 + 0.25 20dB F1 + 0.25 10dB F1',
             'clean_f1_guard':.01,'guard_reference':'existing distilled seed44 FP32 evaluated in FP32',
             'fallback':'if no candidate passes guard, export best experimental candidate without replacing firmware'},
        'quantization':{'method':'static signed INT8 QDQ, per-channel weights, MinMax',
                        'training_calibration':{'clean':128,'20dB':64,'10dB':64},
                        'thresholds':'frozen FP32 clean-validation thresholds'},
        'test_used_for_selection':False,'test_fold_previously_reused':True,
        'normalization':'reuse and verify moments fitted on the 17084 clean training records only',
        'source_code_sha256':digest(ROOT/'src/training/noise.py'),
        'metadata_sha256':digest(Path(config['dataset']['root'])/'ptbxl_database.csv'),
        'protected_artifacts':{str(p.relative_to(ROOT)):digest(p) for p in protected},
    }
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')==protocol, 'Existing experiment protocol changed; use a new experiment directory.'
    else:
        save(OUT/'protocol.json',protocol)
        for name,frame in frames.items():
            frame[['ecg_id','patient_id','strat_fold',*SUPERCLASSES]].to_csv(OUT/f'cohort_{name}.csv',index=False)
    return base,config,frames,protocol


def clean_view(split, base):
    values,frame=get_split(split,base)
    return Path(values.filename),values,frame


def noisy_views(split, base, levels, workers=8):
    clean_path,clean,frame=clean_view(split,base)
    config=FilterConfig.from_mapping(base['config']['signal_processing'])
    root=Path(base['config']['dataset']['root'])
    folder=OUT/'noise_views'
    folder.mkdir(parents=True,exist_ok=True)
    paths={level:folder/f'{split}_snr{level}.npy' for level in levels}
    common={'split':split,'ids_sha256':signature(frame.ecg_id.tolist()),
            'filter':base['config']['signal_processing'],'noise_seed':NOISE_SEED,
            'noise_code_sha256':digest(ROOT/'src/training/noise.py'),'clean_cache_metadata_sha256':digest(clean_path.with_suffix('.json'))}
    missing=[]
    for level,path in paths.items():
        expected=signature({**common,'snr_db':level})
        if path.exists() and path.with_suffix('.json').exists():
            metadata=read(path.with_suffix('.json'))
            assert metadata['fingerprint']==expected and np.load(path,mmap_mode='r').shape==clean.shape
        else:
            assert not path.exists() and not path.with_suffix('.json').exists(), 'Incomplete noise cache.'
            missing.append(level)
    if not missing:
        return [clean_path,*[paths[level] for level in levels]],frame
    arrays={}
    for level in missing:
        temporary=paths[level].with_suffix('.building')
        assert not temporary.exists(), 'Preserve and inspect incomplete noise build before retrying.'
        arrays[level]=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=clean.shape)
    raw_stats={level:[] for level in missing}
    filtered_stats={level:[] for level in missing}
    split_index={'train':1,'validation':2,'test':3}[split]
    rows=list(frame.itertuples(index=False))

    def process(index):
        row=rows[index]
        raw=load_ecg(root,row.filename_lr,sampling_rate_hz=100).signal
        result={}
        for level in missing:
            corrupted,noise=add_raw_gaussian_noise(raw,level,seed=np.random.SeedSequence([NOISE_SEED,split_index,int(row.ecg_id),level]))
            prepared=preprocess_ecg(corrupted,100,config=config).T.copy()
            result[level]=(prepared,snr_db(raw,noise),snr_db(clean[index],prepared-clean[index],axis=-1))
        return index,result

    status(f'Preparing {split} noisy raw ECG views {missing} dB ({len(frame)} records)')
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0,len(frame),128):
            for index,result in pool.map(process,range(start,min(len(frame),start+128))):
                for level,(prepared,raw_snr,filtered_snr) in result.items():
                    assert prepared.shape==(12,1000) and np.isfinite(prepared).all()
                    arrays[level][index]=prepared
                    raw_stats[level].append(raw_snr)
                    filtered_stats[level].append(filtered_snr)
            if start//128 % 8==0 or start+128>=len(frame):
                status(f'{split} noise cache: {min(len(frame),start+128)}/{len(frame)} records, {time.monotonic()-started:.1f}s')
    for level in missing:
        arrays[level].flush()
        del arrays[level]
        paths[level].with_suffix('.building').replace(paths[level])
        raw_values=np.concatenate(raw_stats[level]); filtered_values=np.concatenate(filtered_stats[level])
        valid=raw_values[np.isfinite(raw_values)]
        assert np.max(np.abs(valid-level))<.01
        residual=filtered_values[np.isfinite(filtered_values)]
        save(paths[level].with_suffix('.json'),{**common,'fingerprint':signature({**common,'snr_db':level}),
             'shape':list(clean.shape),'snr_db':level,'stage':'raw mV before preprocessing',
             'measured_raw_snr_range':[float(valid.min()),float(valid.max())],
             'equivalent_postprocessing_residual_snr_quantiles':np.quantile(residual,[0,.1,.5,.9,1]).tolist(),
             'zero_power_lead_count':int((~np.isfinite(raw_values)).sum()),'seconds':time.monotonic()-started})
    return [clean_path,*[paths[level] for level in levels]],frame


def predict(path, views, frame):
    cp=load(path)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(cp['model_type'],dropout=cp['config']['model']['dropout']).to(device).eval()
    model.load_state_dict(cp['model_state'])
    values=[]
    with torch.inference_mode():
        for view in views:
            array=np.load(view,mmap_mode='r')
            scores=[]
            for start in range(0,len(frame),64):
                x=torch.from_numpy(np.array(array[start:start+64],copy=True)).to(device)
                scores.append(torch.sigmoid(model(x)).cpu().numpy())
            values.append(np.concatenate(scores))
    del model
    return values,cp


def train(name, kind, seed, views, frame, validation_view, validation_frame, base, *, noisy, teacher=None):
    folder=OUT/name
    path=folder/f'checkpoint/{kind}_best.pt'
    history_path=folder/'history.json'
    if history_path.exists():
        assert path.exists() and read(history_path)['checkpoint_sha256']==digest(path)
        status(f'Reusing completed {name}')
        return path
    assert not path.exists(), f'Partial training: {name}; inspect before resuming.'
    cfg=copy.deepcopy(base['config'])
    cfg['model']['type']=kind
    cfg['training'].pop('distillation',None)
    cfg['training'].update(random_seed=seed,epochs=30,early_stopping_patience=8)
    for key in ['augmentation_gain','augmentation_noise','mixup_alpha']:
        cfg['training'].pop(key,None)
    cfg['training']['noise_views']={'raw_snr_db':[20,10],'probabilities':[.5,.25,.25] if noisy else [1,0,0]}
    for key,child in [('checkpoint_dir','checkpoint'),('metrics_dir','metrics'),('figure_dir','figure'),('predictions_dir','predictions')]:
        cfg['outputs'][key]=str((folder/child).relative_to(ROOT))
    targets=frame[list(SUPERCLASSES)].to_numpy(dtype=np.float32)
    kwargs={}
    if teacher is not None:
        logits=np.load(teacher['logits'],allow_pickle=False)
        assert str(logits['teacher_sha256'])==digest(teacher['checkpoint'])
        assert np.array_equal(logits['view_fingerprints'],[read(v.with_suffix('.json'))['fingerprint'] for v in views])
        kwargs={'teacher_logits':logits['logits'],'teacher_ids':logits['ecg_ids']}
        cfg['training']['distillation']={'alpha':.5,'temperature':2,'teacher_checkpoint':str(teacher['checkpoint']),
             'teacher_sha256':digest(teacher['checkpoint']),'logits_path':str(teacher['logits']),
             'alignment':'same record AND noise view'}
    save(folder/'config.json',cfg)
    set_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp=device.type=='cuda'
    dataset=NoiseViewsDataset(views,targets,frame.ecg_id,probabilities=[.5,.25,.25] if noisy else [1,0,0],seed=seed,**kwargs)
    valid=NoiseViewsDataset([validation_view],validation_frame[list(SUPERCLASSES)],validation_frame.ecg_id,probabilities=[1],seed=seed)
    loader=DataLoader(dataset,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0,pin_memory=use_amp)
    val_loader=DataLoader(valid,batch_size=32,shuffle=False,num_workers=0,pin_memory=use_amp)
    model=build_model(kind,dropout=float(cfg['model']['dropout'])).to(device)
    positives=targets.sum(0)
    weights=np.divide(len(targets)-positives,positives,out=np.ones(5,dtype=np.float32),where=positives>0)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).to(device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    history=[]; best=-float('inf'); stale=0
    status(f'Training {name}; {len(frame)} records; device={device}; clean/noise={cfg["training"]["noise_views"]["probabilities"]}')
    for epoch in range(1,31):
        dataset.set_epoch(epoch)
        started=time.monotonic()
        loss,_,_=_run_epoch(model,loader,criterion,device,optimizer=optimizer,scaler=scaler,use_amp=use_amp,
                            distillation={'alpha':.5,'temperature':2} if teacher else None)
        val_loss,y,p=_run_epoch(model,val_loader,criterion,device,use_amp=use_amp)
        metrics=calculate_multilabel_metrics(y,p,.5)
        history.append({'epoch':epoch,'training_loss':loss,'validation_loss':val_loss,'clean_validation':metrics,
                        'view_counts':np.bincount(dataset.selected_views,minlength=3).tolist(),
                        'seconds':time.monotonic()-started})
        status(f'{name} epoch {epoch}: AUROC={metrics["macro_auroc"]:.6f}, F1@0.5={metrics["macro_f1"]:.6f}, {history[-1]["seconds"]:.1f}s')
        if metrics['macro_auroc']>best+1e-4:
            best=metrics['macro_auroc']; stale=0
            _save_checkpoint(path,{'model_type':kind,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                'epoch':epoch,'scope':'full','training_records':len(frame),'validation_records':len(validation_frame),
                'validation_macro_auroc':best,'sampling_rate_hz':100,'lead_names':list(LEAD_NAMES),
                'superclasses':list(SUPERCLASSES),'thresholds':[.5]*5,'pos_weight':weights.tolist(),
                'config':cfg,'torch_version':str(torch.__version__),'device':str(device)})
        else:
            stale+=1
            if stale>=8:
                break
    cp=load(path)
    model.load_state_dict(cp['model_state'])
    _,y,p=_run_epoch(model,val_loader,criterion,device,use_amp=use_amp)
    cp['thresholds']=optimize_thresholds(y,p,fallback=.5)
    cp['validation_metrics_tuned']=calculate_multilabel_metrics(y,p,cp['thresholds'])
    _save_checkpoint(path,cp)
    save(history_path,{'history':history,'best_epoch':cp['epoch'],'checkpoint_sha256':digest(path),
                       'parameter_count':sum(parameter.numel() for parameter in model.parameters())})
    del model
    return path


def teacher_targets(teacher, views, frame):
    path=OUT/'teacher_noise_training_logits.npz'
    fingerprints=[read(v.with_suffix('.json'))['fingerprint'] for v in views]
    if path.exists():
        data=np.load(path,allow_pickle=False)
        assert str(data['teacher_sha256'])==digest(teacher)
        np.testing.assert_array_equal(data['ecg_ids'],frame.ecg_id)
        np.testing.assert_array_equal(data['view_fingerprints'],fingerprints)
        return path
    cp=load(teacher); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(cp['model_type'],dropout=cp['config']['model']['dropout']).to(device).eval()
    model.load_state_dict(cp['model_state']); outputs=[]
    status('Generating new teacher logits for clean, 20dB and 10dB TRAINING views')
    with torch.inference_mode():
        for view in views:
            x=np.load(view,mmap_mode='r'); chunks=[]
            for start in range(0,len(frame),64):
                chunks.append(model(torch.from_numpy(np.array(x[start:start+64],copy=True)).to(device)).cpu().numpy())
            outputs.append(np.concatenate(chunks))
    np.savez_compressed(path,logits=np.stack(outputs),ecg_ids=frame.ecg_id.to_numpy(),
                        teacher_sha256=np.asarray(digest(teacher)),view_fingerprints=np.asarray(fingerprints))
    del model
    return path


def model_inventory():
    items=[]
    for arm in ['baseline','distilled']:
        for seed in SEEDS:
            items.append({'name':f'A_{arm}_{seed}','arm':f'A_{arm}','seed':seed,
                          'path':ROOT/f'outputs/distillation/{arm}_seed{seed}/checkpoint/cnn1d_best.pt'})
    for arm in ['teacher_clean','teacher_noise']:
        items.append({'name':arm+'_44','arm':arm,'seed':44,'path':OUT/f'{arm}_44/checkpoint/inception1d_best.pt'})
    for arm in ['B_noise_supervised','B_noise_distilled']:
        for seed in SEEDS:
            items.append({'name':f'{arm}_{seed}','arm':arm,'seed':seed,'path':OUT/f'{arm}_{seed}/checkpoint/cnn1d_best.pt'})
    return items


def evaluate(items,split,views,frame):
    conditions=['clean',*[f'{level}dB' for level in LEVELS[split]]]
    targets=frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    result={}
    for item in items:
        path=OUT/'evaluation'/split/(item['name']+'.json')
        sha=digest(item['path'])
        if path.exists():
            report=read(path)
            assert report['checkpoint_sha256']==sha and report['conditions']==conditions
        else:
            status(f'Evaluating {split}: {item["name"]} on {conditions}')
            scores,cp=predict(item['path'],views,frame)
            report={'checkpoint_sha256':sha,'conditions':conditions,'metrics':{}}
            for condition,p in zip(conditions,scores):
                report['metrics'][condition]=calculate_multilabel_metrics(targets,p,cp['thresholds'])
                target_file=path.parent/f'{item["name"]}_{condition}_predictions.npz'
                target_file.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target_file,ecg_id=frame.ecg_id.to_numpy(),targets=targets,
                                    probabilities=p,thresholds=np.asarray(cp['thresholds']))
            save(path,report)
        result[item['name']]=report['metrics']
    return result


def command(args,log):
    with log.open('w',encoding='utf-8') as file:
        result=subprocess.run([str(arg) for arg in args],cwd=ROOT,stdout=file,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed; inspect {log}')


def export(selected,base,frames,summary):
    folder=OUT/'selected_export'; folder.mkdir(exist_ok=True)
    path=ROOT/selected['checkpoint']
    if not (folder/'deployment_manifest.json').exists():
        status('Exporting selected B distilled student to FP32 ONNX')
        command([sys.executable,ROOT/'scripts/prepare_stm32.py','--checkpoint',path,'--output-dir',folder],folder/'export.log')
    manifest=read(folder/'deployment_manifest.json')
    assert manifest['source_checkpoint_sha256']==digest(path)
    if not manifest['calibration'].get('noise_view_mix'):
        views,_=noisy_views('train',base,LEVELS['train'])
        rows=np.asarray(manifest['calibration']['row_indices'])
        assignment=np.asarray([0]*128+[1]*64+[2]*64)
        np.random.default_rng(NOISE_SEED+100).shuffle(assignment)
        array=np.stack([np.load(views[view],mmap_mode='r')[row] for row,view in zip(rows,assignment)]).astype(np.float32)
        np.save(folder/'calibration_train.npy',array,allow_pickle=False)
        np.savez_compressed(folder/'calibration_train.npz',ecg=array)
        manifest['calibration'].update(noise_view_mix={'clean':128,'20dB':64,'10dB':64},
             selected_views=assignment.tolist(),noise_added_to_raw_mV=True)
        for name in ['calibration_train.npy','calibration_train.npz']:
            manifest['artifacts'][name]={'size_bytes':(folder/name).stat().st_size,'sha256':digest(folder/name)}
        save(folder/'deployment_manifest.json',manifest)
    if not (folder/'int8_validation_report.json').exists():
        status('INT8 QDQ quantization: 256 training-only clean/noisy calibration records')
        command([sys.executable,ROOT/'scripts/quantize_cubeai.py','--export-dir',folder],folder/'quantization.log')
    import onnxruntime as ort
    from scripts.quantize_cubeai import probabilities
    options=ort.SessionOptions(); options.intra_op_num_threads=4; options.inter_op_num_threads=1
    quantized={}
    for name,model,thresholds in [('A_deployed_seed44_INT8',OLD_INT8,load(BASE)['thresholds']),
                                ('B_selected_INT8',folder/'ptbxl_cnn1d_int8.onnx',load(path)['thresholds'])]:
        result_file=folder/(name+'_robustness.json')
        if result_file.exists():
            report=read(result_file)
            assert report['onnx_sha256']==digest(model)
        else:
            session=ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
            report={'onnx_sha256':digest(model),'thresholds':thresholds,'validation':{},'test':{}}
            for split in ['validation','test']:
                views,frame=noisy_views(split,base,LEVELS[split])
                y=frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
                for condition,view in zip(['clean',*[f'{v}dB' for v in LEVELS[split]]],views):
                    status(f'ORT {name}: {split} {condition}')
                    p=probabilities(session,np.load(view,mmap_mode='r'))
                    report[split][condition]=calculate_multilabel_metrics(y,p,thresholds)
                    np.savez_compressed(folder/f'{name}_{split}_{condition}_predictions.npz',
                                        ecg_id=frame.ecg_id.to_numpy(),targets=y,probabilities=p,thresholds=thresholds)
            save(result_file,report)
        quantized[name]=report
    if not (folder/'cubeai_resource.json').exists():
        if CLI.exists():
            status('Cube.AI resource analysis for B INT8 CNN')
            command([CLI,'analyze','--model',folder/'ptbxl_cnn1d_int8.onnx','--name','ecg_noise_b',
              '--target','stm32f4','--c-api','legacy','--compression','none','--optimization','ram',
              '--inputs-ch-position','chfirst','--output-data-type','float32','--quiet',
              '--workspace',folder/'cubeai_workspace','--output',folder/'cubeai_analysis'],folder/'cubeai.log')
            info=read(folder/'cubeai_analysis/ecg_noise_b_c_info.json')
            text=(folder/'cubeai_analysis/ecg_noise_b_analyze_report.txt').read_text(encoding='utf-8')
            resource={'weights_bytes':info['memory_footprint']['weights'],'activations_bytes':info['memory_footprint']['activations'],
                      'macc':int(re.search(r'^macc\s+:\s+([\d,]+)',text,re.M).group(1).replace(',','')),
                      'physical_board_test':False,'complete_firmware_compilation':False}
        else:
            resource={'available':False,'reason':'Cube.AI CLI not found','physical_board_test':False}
        save(folder/'cubeai_resource.json',resource)
    summary.update(quantized=quantized,cubeai_resource=read(folder/'cubeai_resource.json'))
    save(OUT/'summary.json',summary)


def update_documents(summary):
    items=summary['models']; test=summary['test']; selected=summary['selection']['name']
    names=[item['name'] for item in items]
    lines=['## B：真实 ECG 加噪、教师训练、学生蒸馏与 INT8','',
      '日期：2026-10-03。仅高斯白噪声，未生成合成 ECG。原始 mV 上按导联 AC 功率加入噪声，再执行原有基线处理、0.5–40 Hz 带通和训练集全局标准化。',
      '', '训练保留原 17084 条记录，每个 epoch 为各记录选择一个版本：干净 50%、20 dB 25%、10 dB 25%。每条记录预先生成两份固定噪声版本，各轮重新抽取版本。训练、验证和测试使用不同的固定噪声种子，先按官方患者划分，再加噪；加噪版本不增加独立患者数。',
      '', '教师为 Inception：seed44，从零训练干净对照和加噪版本；学生 CNN 加噪普通训练 / 蒸馏各做 42、43、44 三个种子。蒸馏 alpha=0.5、T=2；教师缓存含三个输入版本，按 ECG ID 与版本同时对齐。旧普通 / 蒸馏学生三种子作为历史对照。教师与普通对照的统计均值、缓存指纹及原部署文件哈希均有记录。',
      '', '所有训练最多 30 轮、早停 8 轮，按干净验证 AUROC 选权重；阈值只用干净验证数据调整。部署候选在测试前冻结，以干净/20/10 dB 验证 F1 的 0.5/0.25/0.25 加权分数选取，先检查干净验证 F1 相对原 seed44 是否下降超过 0.01。',
      '',f'选出的蒸馏候选：`{selected}`。是否通过干净验证保护条件：`{summary["selection"]["clean_guard_passed"]}`。',
      '', '### 各模型测试整体指标','', '| 模型 | 条件 | Macro AUROC | Precision | Recall | F1 |', '|---|---|---:|---:|---:|---:|']
    for name in names:
        for condition,m in test[name].items():
            lines.append(f'| {name} | {condition} | '+' | '.join(f'{m[k]:.6f}' for k in ['macro_auroc','macro_precision','macro_recall','macro_f1'])+' |')
    if 'quantized' in summary:
        lines+=['','### 当前部署模型与 B 候选 INT8','', '| 模型 | 条件 | Macro AUROC | Precision | Recall | F1 |', '|---|---|---:|---:|---:|---:|']
        for name,result in summary['quantized'].items():
            for condition,m in result['test'].items():
                lines.append(f'| {name} | {condition} | '+' | '.join(f'{m[k]:.6f}' for k in ['macro_auroc','macro_precision','macro_recall','macro_f1'])+' |')
        lines+=['','### INT8 逐类结果','', '| 模型 | 条件 | 类别 | AUROC | Precision | Recall | F1 |','|---|---|---|---:|---:|---:|---:|']
        for name,result in summary['quantized'].items():
            for condition,m in result['test'].items():
                for label,c in m['per_class'].items():
                    lines.append(f'| {name} | {condition} | {label} | '+' | '.join(f'{c[k]:.6f}' for k in ['auroc','precision','recall','f1'])+' |')
        resource=summary['cubeai_resource']
        lines+=['',f'Cube.AI 模型资源：`{json.dumps(resource,ensure_ascii=False)}`。模型资源分析不能代替完整固件链接或板上耗时。']
    lines+=['','### 证据与边界','',
        '完整整体和逐类结果、逐条预测、配置、训练历史、固定噪声缓存及实际 SNR 审计位于 `outputs/noise_robustness_b/`。`protocol.json` 在训练前写入，`frozen_selection.json` 在测试前写入；`summary.json` 保存全部结果。`selected_export/` 为新候选 FP32 / INT8 ONNX，原 seed44 固件保留。',
        '', '本轮原始 SNR 与滤波后等效残差 SNR 分别记录，不能把 20 dB 原始噪声称为滤波后仍恰好 20 dB。验证 / 测试噪声是固定的受控高斯噪声，不能据此证明对肌电、电极运动或外部数据库的泛化。教师仅一个种子；历史蒸馏教师曾使用原始含全零标签训练范围，与本轮筛选后教师不同，蒸馏变化不应全部归因于噪声。测试折此前已多次复用，仍不属于新的盲测。',
        '', '复现：`.venv/Scripts/python.exe scripts/run_noise_robustness.py`。只跑准备或训练可用 `--phase prepare` / `--phase train`；已完成且校验一致的阶段可复用，不允许静默修改原实验协议。']
    overall_index=lines.index('### 各模型测试整体指标')
    quantized_index=lines.index('### 当前部署模型与 B 候选 INT8') if 'quantized' in summary else None
    evidence_index=lines.index('### 证据与边界')
    statistics={}
    statistics_lines=['### 三个随机种子的对照统计','',
      '以下为测试 Macro F1 的均值 ± 样本标准差（ddof=1），仅有三个种子，不代表统计显著性。普通 CNN 的 A→B 对照用于判断加噪影响，B 普通→B 蒸馏用于判断本轮蒸馏影响。历史 A 蒸馏与 B 蒸馏使用不同教师。',
      '', '| 训练方案 | 干净 | 30 dB | 20 dB | 10 dB |', '|---|---:|---:|---:|---:|']
    conditions=['clean','30dB','20dB','10dB']
    for arm in ['A_baseline','A_distilled','B_noise_supervised','B_noise_distilled']:
        statistics[arm]={}
        values=[]
        for condition in conditions:
            statistics[arm][condition]={}
            for key in ['macro_auroc','macro_precision','macro_recall','macro_f1']:
                sample=np.asarray([test[f'{arm}_{seed}'][condition][key] for seed in SEEDS])
                statistics[arm][condition][key]={'mean':float(sample.mean()),'sample_std':float(sample.std(ddof=1)),
                                               'seeds':SEEDS,'values':sample.tolist()}
            m=statistics[arm][condition]['macro_f1']
            values.append(f'{m["mean"]:.6f} ± {m["sample_std"]:.6f}')
        statistics_lines.append(f'| {arm} | '+' | '.join(values)+' |')
    statistics['paired_deltas']={}
    statistics_lines+=['','| 成对比较（后者减前者） | 干净 ΔF1 | 30 dB ΔF1 | 20 dB ΔF1 | 10 dB ΔF1 |',
                       '|---|---:|---:|---:|---:|']
    for first,second in [('A_baseline','B_noise_supervised'),('B_noise_supervised','B_noise_distilled')]:
        name=f'{first} → {second}'
        statistics['paired_deltas'][name]={}
        values=[]
        for condition in conditions:
            differences=np.asarray([test[f'{second}_{seed}'][condition]['macro_f1']-
                                    test[f'{first}_{seed}'][condition]['macro_f1'] for seed in SEEDS])
            statistics['paired_deltas'][name][condition]={'mean':float(differences.mean()),
                       'sample_std':float(differences.std(ddof=1)), 'seeds':SEEDS,'values':differences.tolist()}
            values.append(f'{differences.mean():+.6f} ± {differences.std(ddof=1):.6f}')
        statistics_lines.append(f'| {name} | '+' | '.join(values)+' |')
    save(OUT/'seed_statistics.json',statistics)
    method=lines[:overall_index]+statistics_lines+['']
    delta_noise=statistics['paired_deltas']['A_baseline → B_noise_supervised']
    delta_kd=statistics['paired_deltas']['B_noise_supervised → B_noise_distilled']
    method+=['### 本轮结果','',
      f'普通 CNN 加噪后，三个种子的平均 F1 在干净测试集变化 {delta_noise["clean"]["mean"]:+.6f}，在 10 dB 测试集变化 {delta_noise["10dB"]["mean"]:+.6f}。加噪蒸馏相对同批加噪普通训练，干净 / 10 dB 的平均 F1 变化为 {delta_kd["clean"]["mean"]:+.6f} / {delta_kd["10dB"]["mean"]:+.6f}。',
      '', '应分别判断加噪与蒸馏的收益，不能把某个指标提高解释成所有指标都提高。B 的蒸馏候选只在预先限定的三种蒸馏学生中按验证集选择，测试结果不用于重新挑选候选。','']
    method+=['### 阅读结果','',
      'Macro Precision / Recall / F1 均为五类独立指标的算术平均。所有历史和新增 FP32 模型本轮统一用 PyTorch FP32 重新推理，旧报告采用混合精度推理时可能出现极小差异；没有修改历史模型权重或阈值。所有噪声条件沿用各自干净验证阈值，没有按测试条件重新调阈值。', '']
    if quantized_index is not None:
        method+=lines[quantized_index:lines.index('### INT8 逐类结果')]
        method+=['Cube.AI 模型资源：`'+json.dumps(summary['cubeai_resource'],ensure_ascii=False)+'`。这里只完成模型资源分析，完整固件和板上运行尚未验证。','']
        old=summary['quantized']['A_deployed_seed44_INT8']['test']
        new=summary['quantized']['B_selected_INT8']['test']
        changes={condition:round(new[condition]['macro_f1']-old[condition]['macro_f1'],6) for condition in conditions}
        method+=['相对当前部署 INT8，B 候选的测试 Macro F1 变化为：`'+json.dumps(changes,ensure_ascii=False)+'`。候选按验证集选择，不能因为某个种子在测试集更高而事后替换。','']
        if changes['clean']<0:
            method+=['B 蒸馏 INT8 的干净测试 F1 低于当前 seed44，因此本轮仅保存为实验候选，现有 seed44 模型、HEX 和串口程序继续保留。','']
    method+=lines[evidence_index:]
    method+=['','整体与逐类明细见 [各模型整体与逐类指标总表](各模型整体与逐类指标总表.md)；三种子四项指标统计见 `outputs/noise_robustness_b/seed_statistics.json`。']
    if (OUT/'audit.json').exists():
        audit=read(OUT/'audit.json')
        method+=['',f'独立复核：{audit["prediction_files_recomputed"]} 份逐条预测的 AUROC、Precision、Recall、F1、阳性样本数和混淆矩阵全部与报告一致；患者隔离、全零标签剔除、固定阈值、验证集选模、训练集量化校准和原部署哈希检查均通过。见 `outputs/noise_robustness_b/audit.json`。']
    full=lines[:overall_index]+statistics_lines+['']+lines[overall_index:evidence_index]
    full+=['### 所有 FP32 模型逐类测试指标','',
           '| 模型 | 条件 | 类别 | 样本阳性数 | AUROC | Precision | Recall | F1 |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for name in names:
        for condition,m in test[name].items():
            for label,c in m['per_class'].items():
                full.append(f'| {name} | {condition} | {label} | {c["support"]} | '+
                            ' | '.join(f'{c[k]:.6f}' for k in ['auroc','precision','recall','f1'])+' |')
    full+=['']+lines[evidence_index:]
    for file,content in [('md/模型训练与优化.md',method),('md/各模型整体与逐类指标总表.md',full)]:
        block='\n'.join(content)+'\n'
        path=ROOT/file; text=path.read_text(encoding='utf-8')
        start='<!-- NOISE_B_START -->'; end='<!-- NOISE_B_END -->'
        if start in text:
            text=text.split(start)[0]+start+'\n'+block+end+text.split(end,1)[1]
        else:
            text=text.rstrip()+'\n\n'+start+'\n'+block+end+'\n'
        path.write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for name,result in summary.get('quantized',{}).items():
        order=['clean','30dB','20dB','10dB']
        ax.plot(order,[result['test'][condition]['macro_f1'] for condition in order],marker='o',label=name)
    ax.set_ylabel('Macro F1'); ax.set_xlabel('Raw-input Gaussian-noise condition')
    ax.set_title('PTB-XL filtered fold 10: fixed thresholds'); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'int8_noise_comparison.png',dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','prepare','train','evaluate','export','report'],default='all')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    base,cfg,frames,protocol=initialize()
    if args.phase in ['all','prepare','train']:
        raw_settings={**cfg['signal_processing'],'normalization':'none'}
        raw_settings.pop('training_mean',None); raw_settings.pop('training_std',None)
        raw=ensure_preprocessed_cache(frames['train'],cfg['dataset']['root'],ROOT/cfg['outputs']['cache_dir'],
               split_name='train_full',sampling_rate_hz=100,config=FilterConfig.from_mapping(raw_settings))
        mean,std=fit_training_normalization(raw)
        np.testing.assert_allclose(mean,cfg['signal_processing']['training_mean'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(std,cfg['signal_processing']['training_std'],rtol=0,atol=1e-12)
        views,frame=noisy_views('train',base,LEVELS['train'])
        val_views,val_frame=noisy_views('validation',base,LEVELS['validation'])
        save(OUT/'normalization_audit.json',{'training_records':len(frame),'matches_existing_students':True,
                                          'mean':mean,'std':std,'validation_or_test_used':False})
        if args.phase=='prepare':
            status('Noise preparation complete')
            return
        train('teacher_clean_44','inception1d',44,views,frame,val_views[0],val_frame,base,noisy=False)
        teacher=train('teacher_noise_44','inception1d',44,views,frame,val_views[0],val_frame,base,noisy=True)
        logits=teacher_targets(teacher,views,frame)
        for seed in SEEDS:
            train(f'B_noise_supervised_{seed}','cnn1d',seed,views,frame,val_views[0],val_frame,base,noisy=True)
            train(f'B_noise_distilled_{seed}','cnn1d',seed,views,frame,val_views[0],val_frame,base,noisy=True,
                  teacher={'checkpoint':teacher,'logits':logits})
        if args.phase=='train':
            status('Eight training runs complete; test split has not been evaluated')
            return
    items=model_inventory()
    if args.phase in ['all','evaluate']:
        val_views,val_frame=noisy_views('validation',base,LEVELS['validation'])
        validation=evaluate(items,'validation',val_views,val_frame)
        reference=validation['A_distilled_44']['clean']['macro_f1']
        candidates=[]
        for item in items:
            if item['arm']=='B_noise_distilled':
                m=validation[item['name']]
                score=.5*m['clean']['macro_f1']+.25*m['20dB']['macro_f1']+.25*m['10dB']['macro_f1']
                candidates.append({'name':item['name'],'seed':item['seed'],'checkpoint':str(item['path'].relative_to(ROOT)),
                    'checkpoint_sha256':digest(item['path']),'validation_score':score,
                    'clean_guard_passed':m['clean']['macro_f1']>=reference-.01,'clean_validation_f1':m['clean']['macro_f1']})
        eligible=[c for c in candidates if c['clean_guard_passed']]
        selected=max(eligible or candidates,key=lambda c:(c['validation_score'],c['clean_validation_f1']))
        frozen={'selected':selected,'all_candidates':candidates,'protocol_sha256':digest(OUT/'protocol.json'),
                'test_used_for_selection':False,'frozen_at':datetime.datetime.now(TZ).isoformat()}
        if (OUT/'frozen_selection.json').exists():
            assert read(OUT/'frozen_selection.json')['selected']==selected
        else:
            save(OUT/'frozen_selection.json',frozen)
        status(f'Validation selection frozen: {selected["name"]}; now preparing and evaluating TEST noise')
        test_views,test_frame=noisy_views('test',base,LEVELS['test'])
        test=evaluate(items,'test',test_views,test_frame)
        summary={'protocol':protocol,'selection':selected,'models':[{**item,'path':str(item['path'].relative_to(ROOT))} for item in items],
                 'validation':validation,'test':test,'original_artifacts_preserved':True}
        save(OUT/'summary.json',summary)
        if args.phase=='evaluate':
            update_documents(summary)
            status('FP32 B evaluation and documentation complete')
            return
    summary=read(OUT/'summary.json')
    if args.phase in ['all','export']:
        assert (OUT/'frozen_selection.json').exists()
        export(summary['selection'],base,frames,summary)
    for name,sha in protocol['protected_artifacts'].items():
        assert digest(ROOT/name)==sha, f'Original deployment or baseline changed: {name}'
    update_documents(read(OUT/'summary.json'))
    status('B experiment complete: noise training, distillation, INT8, validation/test reports and existing MD updated')


if __name__=='__main__':
    main()
