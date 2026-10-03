"""Full external cohort audit and frozen-model evaluation, with explicit masks."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import wfdb
from scipy.special import expit
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_external_chapman import find_dataset_root, _load_and_preprocess
from src.data.ptbxl import SUPERCLASSES, LEAD_NAMES
from src.external_validation.chapman import parse_diagnosis_codes
from src.external_validation.chapman_audit import (
    MAPPING_VERSION, FOUR_CLASSES, encode_audited_labels, load_catalog, rule_for,
)
from src.inference import load_predictor
from src.signal_processing.filters import normalize_signal

OUT = ROOT / 'outputs/external_validation/chapman_audited'


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024 * 1024):
            sha.update(data)
    return sha.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def summarize_metrics(targets, masks, probabilities, thresholds, *, mode):
    per_class = {}
    for index, name in enumerate(SUPERCLASSES):
        selected = masks[:, index]
        truth = targets[selected, index]
        scores = probabilities[selected, index]
        predicted = scores >= thresholds[index]
        positives = int(truth.sum())
        negatives = int(len(truth) - positives)
        tp = int(np.sum(predicted & (truth == 1)))
        fn = positives - tp
        # Wilson interval communicates finite positive sample uncertainty,
        # without using any unknown record as a negative.
        recall_interval = None
        if positives:
            observed = tp / positives
            z = 1.959963984540054
            denominator = 1 + z * z / positives
            center = (observed + z * z / (2 * positives)) / denominator
            half = z * np.sqrt(observed * (1 - observed) / positives + z * z / (4 * positives * positives)) / denominator
            recall_interval = [float(max(0, center - half)), float(min(1, center + half))]
        row = {
            'positive_records': positives, 'negative_records': negatives,
            'unknown_records': int((~selected).sum()),
            'threshold': float(thresholds[index]),
            'true_positive': tp, 'false_negative': fn,
            'recall': tp / positives if positives else None,
            'positive_recall_95ci_wilson': recall_interval,
            'auroc': None, 'precision': None, 'f1': None,
            'confusion_matrix_tn_fp_fn_tp': None,
            'positive_score_median': float(np.median(scores[truth == 1])) if positives else None,
            'positive_score_max': float(np.max(scores[truth == 1])) if positives else None,
        }
        if mode == 'conditional_absence_negative' and positives and negatives:
            precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average='binary', zero_division=0)
            row.update(auroc=float(roc_auc_score(truth, scores)), precision=float(precision),
                       recall=float(recall), f1=float(f1),
                       confusion_matrix_tn_fp_fn_tp=confusion_matrix(truth, predicted, labels=[0, 1]).ravel().tolist())
        per_class[name] = row
    result = {'label_policy': mode, 'per_class': per_class, 'NORM_scored': False,
              'negative_truth_clinically_verified': False,
              'ptbxl_five_class_macro_directly_comparable': False}
    for metric in ('auroc', 'precision', 'recall', 'f1'):
        values = [per_class[c][metric] for c in FOUR_CLASSES if per_class[c][metric] is not None]
        result['macro_' + metric] = float(np.mean(values)) if values else None
    return result


def audit_headers(dataset, headers, catalog):
    def read_header(header):
        record = header.with_suffix('')
        row = {'record': record.relative_to(dataset).as_posix()}
        codes = ()
        try:
            item = wfdb.rdheader(str(record))
            codes = parse_diagnosis_codes(item.comments or [])
            y, strict, conditional, reasons = encode_audited_labels(codes)
            row.update(codes=list(codes), targets=y.tolist(), positive_only=strict.tolist(),
                       conditional=conditional.tolist(), mapping_reasons=list(reasons),
                       fs=float(item.fs), samples=item.sig_len,
                       lead_names=item.sig_name, units=item.units,
                       comments=item.comments)
            names = [str(name).casefold() for name in item.sig_name]
            if item.fs != 500 or item.sig_len != 5000 or len(names) != 12 or set(names) != {name.casefold() for name in LEAD_NAMES}:
                raise ValueError('Input spec is not 12 standard leads / 500 Hz / 5000 samples')
            row['header_error'] = None
        except Exception as error:
            row['header_error'] = f'{type(error).__name__}: {error}'
            row.setdefault('targets', [0] * 5)
            row['positive_only'] = [False] * 5
            row['conditional'] = [False] * 5
        return row, codes
    rows, count_codes, error_counts = [], Counter(), Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for i, (row, codes) in enumerate(pool.map(read_header, headers), 1):
            count_codes.update(codes)
            if row['header_error']:
                error_counts[row['header_error']] += 1
            rows.append(row)
            if i % 5000 == 0:
                print(f'Audited {i:,}/{len(headers):,} headers', flush=True)
    codes = []
    for code, count in count_codes.most_common():
        role, affected = rule_for(code)
        codes.append({'code': code, 'record_count': count, 'role': role,
                      'affected_classes': list(affected), **catalog.get(code, {'names': [], 'acronyms': [], 'sources': []})})
    save_json(OUT / 'record_label_audit.json', rows)
    save_json(OUT / 'code_audit.json', {'mapping_version': MAPPING_VERSION, 'codes': codes, 'header_errors': dict(error_counts)})
    save_json(OUT / 'header_audit_provenance.json', {
        'mapping_sha256': digest(ROOT / 'src/external_validation/chapman_audit.py'),
        'catalog_sha256': hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
        'manifest_sha256': digest(dataset / 'SHA256SUMS.txt'),
        'record_audit_sha256': digest(OUT / 'record_label_audit.json'),
    })
    return rows


def normalize_batch(base, config):
    """base: [batch, leads, time], filtered physical mV; exactly checkpoint normalization."""
    if config.normalization == 'train_global':
        mean = np.asarray(config.training_mean, dtype=np.float64)[None, :, None]
        std = np.asarray(config.training_std, dtype=np.float64)[None, :, None]
        result = ((base - mean) / std).astype(np.float32)
    else:
        result = np.stack([normalize_signal(row.T, config.normalization).T for row in base])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite normalized batch')
    return np.ascontiguousarray(result, dtype=np.float32)


def prepare_cache(dataset, rows, base_config, workers):
    settings = {'record_audit_sha256': digest(OUT / 'record_label_audit.json'),
                'base_filter': asdict(base_config), 'shape': [len(rows), 12, 1000],
                'dataset_checksum_manifest_sha256': digest(dataset / 'SHA256SUMS.txt')}
    cache = OUT / 'filtered_mv.npy'
    manifest = OUT / 'cache_settings.json'
    ready = OUT / 'cache_completed.npy'
    valid_path = OUT / 'cache_valid.npy'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != settings:
        raise RuntimeError('Cache provenance changed; use another output directory')
    if not manifest.exists():
        save_json(manifest, settings)
        x = np.lib.format.open_memmap(cache, mode='w+', dtype=np.float32, shape=(len(rows), 12, 1000))
        completed = np.zeros(len(rows), dtype=bool)
        valid = np.zeros(len(rows), dtype=bool)
    else:
        x = np.load(cache, mmap_mode='r+')
        completed = np.load(ready) if ready.exists() else np.zeros(len(rows), dtype=bool)
        valid = np.load(valid_path) if valid_path.exists() else np.zeros(len(rows), dtype=bool)
    errors_path = OUT / 'waveform_errors.json'
    errors = json.loads(errors_path.read_text(encoding='utf-8')) if errors_path.exists() else []

    def process(index):
        row = rows[index]
        try:
            if row['header_error']:
                raise ValueError(row['header_error'])
            signal = _load_and_preprocess(dataset / row['record'], base_config)
            if not np.isfinite(signal).all():
                raise ValueError('Nonfinite filtered waveform')
            return index, signal, None
        except Exception as error:
            return index, None, f'{type(error).__name__}: {error}'

    pending = np.flatnonzero(~completed).tolist()
    print(f'Preparing {len(pending):,} uncached ECGs with {workers} workers', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for step, (index, signal, error) in enumerate(pool.map(process, pending), 1):
            if error:
                errors.append({'record': rows[index]['record'], 'error': error})
                valid[index] = False
                x[index] = 0
            else:
                x[index] = signal
                valid[index] = True
            completed[index] = True
            if step % 1000 == 0:
                x.flush()
                np.save(ready, completed)
                np.save(valid_path, valid)
                save_json(errors_path, errors)
                print(f'Waveforms completed {int(completed.sum()):,}/{len(rows):,}; valid {int(valid.sum()):,}', flush=True)
    x.flush()
    np.save(ready, completed)
    np.save(valid_path, valid)
    save_json(errors_path, errors)
    if not completed.all():
        raise RuntimeError('Unprocessed rows remain')
    return x, valid, errors


def write_report(summary):
    lines = ['# Chapman–Shaoxing–Ningbo：冻结模型外部验证', '',
             '本轮未训练、微调模型，也未使用外部数据调整阈值。NORM 没有可靠真值，不计分。', '',
             f"全量头文件 {summary['headers_scanned']} 条；成功读取波形 {summary['waveforms_evaluated']} 条；排除 {summary['waveforms_excluded']} 条。",
             f"数据源：[PhysioNet ecg-arrhythmia 1.0.0](https://physionet.org/content/ecg-arrhythmia/1.0.0/)。官方 SHA-256 文件校验通过；本地标记：`{summary['checksum_verification_stamp']}`。", '',
             '## 输入与标签口径', '',
             '十二导联、500 Hz、10 秒；依 header 转为 mV，抗混叠重采样到 100 Hz，统一导联顺序，使用每个检查点冻结的滤波与归一化参数，输入 `[1,12,1000]`。缓存的是归一化前物理 mV，未拟合外部均值/方差。', '',
             '主结果只使用明确阳性，报告检出率/召回率。其余未标出的诊断保持未知，因此没有足够的已验证阴性来报告主结果 AUC、Precision 或 F1。', '',
             '附加结果采用“诊断列表完整、未标出的类别为阴性”假设，只用于有形态学诊断的记录；歧义代码按类屏蔽，未审计代码屏蔽其余类别，明确阳性保留。该假设未获得逐类临床阴性验证，相关 AUC/F1 不能称为已证实的外部泛化能力。', '',
             '窦性心律不补为 NORM；高电压不补为 HYP；异常 Q 波不补为 MI；早期复极不直接补为 STTC。', '',
             '## 样本覆盖', '', '| 类别 | 明确阳性 | 假设阴性 | 条件口径未知 |', '|---|---:|---:|---:|']
    reference = summary['results'][0]['conditional_metrics']['per_class']
    for name in SUPERCLASSES:
        m = reference[name]
        lines.append(f"| {name} | {m['positive_records']} | {m['negative_records']} | {m['unknown_records']} |")
    lines += ['', '## 主结果：明确阳性检出率', '', '| 模型 | MI Recall | STTC Recall | CD Recall | HYP Recall | 四类平均 Recall |', '|---|---:|---:|---:|---:|---:|']
    fmt = lambda value: '不可计算' if value is None else f'{value:.4f}'
    for row in summary['results']:
        m = row['positive_only_metrics']
        lines.append('| ' + row['name'] + ' | ' + ' | '.join(fmt(m['per_class'][c]['recall']) for c in FOUR_CLASSES) + ' | ' + fmt(m['macro_recall']) + ' |')
    lines += ['', '## 附加结果：依赖阴性假设的四类指标', '', '| 模型 | AUC | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    for row in summary['results']:
        m = row['conditional_metrics']
        lines.append('| ' + row['name'] + ' | ' + ' | '.join(fmt(m['macro_' + key]) for key in ('auroc', 'precision', 'recall', 'f1')) + ' |')
    by_name = {row['name']: row for row in summary['results']}
    student_fp = by_name['distilled_seed44_float32']
    student_int = by_name['distilled_seed44_int8']
    baseline = by_name['baseline_seed42_float32']
    quant_recall_delta = student_int['positive_only_metrics']['macro_recall'] - student_fp['positive_only_metrics']['macro_recall']
    lines += ['', '## 结果解读', '',
              f'蒸馏学生 INT8 相对同一学生 FP32 的明确阳性平均 Recall 变化为 {quant_recall_delta * 100:+.2f} 个百分点。该差异来自同一输入、同一标签与同一阈值下的量化版本对比。']
    fp_f1 = student_fp['conditional_metrics']['macro_f1']
    int_f1 = student_int['conditional_metrics']['macro_f1']
    baseline_f1 = baseline['conditional_metrics']['macro_f1']
    if fp_f1 is not None and int_f1 is not None and baseline_f1 is not None:
        lines += ['', f'在阴性假设口径下，INT8 相对 FP32 的四类 F1 变化为 {(int_f1 - fp_f1) * 100:+.2f} 个百分点；蒸馏 FP32 相对普通学生 FP32 为 {(fp_f1 - baseline_f1) * 100:+.2f} 个百分点。这里比较的是之前已按 PTB-XL 验证集选择的不同种子模型，不是同种子控制实验，不能把全部差异归因于蒸馏。']
    weakest = min(FOUR_CLASSES, key=lambda c: student_int['positive_only_metrics']['per_class'][c]['recall'] if student_int['positive_only_metrics']['per_class'][c]['recall'] is not None else float('inf'))
    weak = student_int['positive_only_metrics']['per_class'][weakest]
    lines += ['', f"当前部署学生明确阳性检出率最低的类别为 {weakest}：Recall {fmt(weak['recall'])}，检出 {weak['true_positive']}/{weak['positive_records']}，漏检 {weak['false_negative']}。阴性假设不能改变这些明确阳性上的漏检数。",
              '', '本次是另一来源上的验证，不能仅凭输入形状相同认为分布相同；也不能用一次外部结果证明临床适用性。若进一步改阈值或训练，需要另留独立外部测试集。']
    for row in summary['results']:
        lines += ['', f"### {row['name']}", '', '| 类别 | AUC | Precision | Recall | F1 | TP | FN |', '|---|---:|---:|---:|---:|---:|---:|']
        for name in FOUR_CLASSES:
            m = row['conditional_metrics']['per_class'][name]
            lines.append('| ' + name + ' | ' + ' | '.join(fmt(m[k]) for k in ('auroc', 'precision', 'recall', 'f1')) + f" | {m['true_positive']} | {m['false_negative']} |")
    lines += ['', '## 验证与限制', '',
              '- 检查点与量化模型 SHA-256 在前后检查，未改变；阈值来自原 PTB-XL 验证集。',
              '- 所有模型使用相同成功读取记录和标签掩码；保存逐条概率、目标、掩码和阈值，可以独立重算指标。',
              '- 缓存拆分滤波/归一化的结果与直接检查点预处理逐样本核对。',
              '- 外部数据不参加训练、阈值选择或模型选优；不得因本次结果再选择模型并把同一数据称为独立测试。',
              '- 与 PTB-XL 五类宏平均不直接可比，也不等于板上实测。此次为主机端冻结 FP32 / INT8 模型验证。',
              '- 未逐条人工复核临床诊断，未验证源数据与 PTB-XL 患者身份是否重合，未按医院划分结果。',
              '', '## 可复现产物', '',
              '`outputs/external_validation/chapman_audited/summary.json`、`record_label_audit.json`、`code_audit.json`、`waveform_errors.json` 及每个模型的 `*_predictions.npz`。',
              '`scripts/evaluate_audited_chapman.py` 为评估入口；`src/external_validation/chapman_audit.py` 为版本化映射，逐条代码名称保存在审计文件中。']
    (ROOT / 'outputs/reports').mkdir(parents=True, exist_ok=True)
    (ROOT / 'outputs/reports/Chapman_冻结模型外部验证总结.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'outputs/external_validation/chapman_data')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--await-download', action='store_true', help='Wait for the full official download verification stamp')
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error('workers and batch-size must be positive')
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = find_dataset_root(args.dataset_root)
    stamp = dataset / '.codex_sha256_verified'
    if args.await_download:
        began = time.monotonic()
        while not stamp.is_file():
            if time.monotonic() - began > 3600:
                raise RuntimeError('Download verification did not finish in one hour')
            time.sleep(10)
    if not stamp.is_file():
        raise RuntimeError('Official checksum verification required before evaluation')
    headers = sorted((dataset / 'WFDBRecords').rglob('*.hea'))
    if len(headers) != 45152:
        raise RuntimeError(f'Full cohort expected; found {len(headers)} headers')
    dictionary_files = [dataset / 'ConditionNames_SNOMED-CT.csv',
                        ROOT / 'outputs/dataset_compatibility/challenge_scored.csv',
                        ROOT / 'outputs/dataset_compatibility/challenge_unscored.csv']
    catalog = load_catalog(dictionary_files)
    provenance_path = OUT / 'header_audit_provenance.json'
    reusable = False
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        reusable = provenance == {
            'mapping_sha256': digest(ROOT / 'src/external_validation/chapman_audit.py'),
            'catalog_sha256': hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
            'manifest_sha256': digest(dataset / 'SHA256SUMS.txt'),
            'record_audit_sha256': digest(OUT / 'record_label_audit.json'),
        }
    if reusable:
        rows = json.loads((OUT / 'record_label_audit.json').read_text(encoding='utf-8'))
        assert [row['record'] for row in rows] == [p.with_suffix('').relative_to(dataset).as_posix() for p in headers]
        print('Reusing full header audit with matching provenance', flush=True)
    else:
        rows = audit_headers(dataset, headers, catalog)
    paths = [
        ('baseline_seed42', ROOT / 'outputs/distillation/baseline_seed42/checkpoint/cnn1d_best.pt'),
        ('distilled_seed44', ROOT / 'outputs/distillation/distilled_seed44/checkpoint/cnn1d_best.pt'),
        ('teacher_inception', ROOT / 'outputs/optimization/optimized_best.pt'),
        ('old_deployed_resnet', ROOT / 'outputs/optimization/01_global_norm/checkpoint/resnet1d_best.pt'),
        ('original_cnn', ROOT / 'outputs/checkpoints/baseline_best.pt'),
    ]
    predictors = [(name, path, load_predictor(path), digest(path)) for name, path in paths]
    base_configs = [replace(p.filter_config, normalization='none', training_mean=(), training_std=()) for _, _, p, _ in predictors]
    if any(c != base_configs[0] for c in base_configs) or any(p.sampling_rate_hz != 100 for _, _, p, _ in predictors):
        raise RuntimeError('Shared physical filter cache is invalid for these checkpoints')
    base, valid, errors = prepare_cache(dataset, rows, base_configs[0], args.workers)
    indices = np.flatnonzero(valid)
    targets = np.array([row['targets'] for row in rows], dtype=np.uint8)[valid]
    primary = np.array([row['positive_only'] for row in rows], dtype=bool)[valid]
    conditional = np.array([row['conditional'] for row in rows], dtype=bool)[valid]
    record_ids = np.array([row['record'] for row in rows])[valid]
    if len(indices) == 0:
        raise RuntimeError('No valid waveform')
    # Verify optimization before inference, including separate normalization per checkpoint.
    preprocess_checks = []
    check_indices = indices[np.linspace(0, len(indices) - 1, min(10, len(indices)), dtype=int)]
    for name, _, p, _ in predictors:
        maximum = 0.0
        for index in check_indices:
            direct = _load_and_preprocess(dataset / rows[index]['record'], p.filter_config)
            cached = normalize_batch(base[index:index + 1], p.filter_config)[0]
            maximum = max(maximum, float(np.max(np.abs(direct - cached))))
        if maximum > 1e-6:
            raise RuntimeError(f'Preprocessing mismatch for {name}: {maximum}')
        preprocess_checks.append({'model': name, 'samples': len(check_indices), 'max_abs_error': maximum})
    results = []
    model_hashes = {}
    for name, path, predictor, before in predictors:
        artifacts = [('float32', path, before)]
        if name == 'distilled_seed44':
            quant = ROOT / 'outputs/distillation/distilled_export/ptbxl_cnn1d_int8.onnx'
            artifacts.append(('int8', quant, digest(quant)))
        for version, artifact, artifact_hash in artifacts:
            print(f'Evaluating {name}_{version}; n={len(indices):,}; frozen thresholds', flush=True)
            stem = f'{name}_{version}'
            output = OUT / f'{stem}_predictions.npz'
            metadata = {'checkpoint_sha256': before, 'artifact_sha256': artifact_hash,
                        'record_audit_sha256': digest(OUT / 'record_label_audit.json'),
                        'mapping_source_sha256': digest(ROOT / 'src/external_validation/chapman_audit.py'),
                        'preprocessing_source_sha256': digest(ROOT / 'src/signal_processing/filters.py'),
                        'cache_settings_sha256': digest(OUT / 'cache_settings.json')}
            meta_path = OUT / f'{stem}_provenance.json'
            if output.exists() and meta_path.exists() and json.loads(meta_path.read_text(encoding='utf-8')) == metadata:
                with np.load(output) as saved:
                    assert np.array_equal(saved['record_ids'], record_ids)
                    assert np.array_equal(saved['targets'], targets)
                    assert np.array_equal(saved['positive_only_mask'], primary)
                    assert np.array_equal(saved['conditional_mask'], conditional)
                    assert np.array_equal(saved['thresholds'], predictor.thresholds)
                    probabilities = saved['probabilities']
            else:
                probabilities = np.empty((len(indices), 5), dtype=np.float32)
                if version == 'int8':
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 4
                    options.inter_op_num_threads = 1
                    session = ort.InferenceSession(str(artifact), options, providers=['CPUExecutionProvider'])
                    input_name = session.get_inputs()[0].name
                for start in range(0, len(indices), args.batch_size):
                    selected = indices[start:start + args.batch_size]
                    inputs = normalize_batch(base[selected], predictor.filter_config)
                    if version == 'float32':
                        with torch.inference_mode():
                            batch = torch.from_numpy(inputs).to(predictor.device)
                            scores = torch.sigmoid(predictor.model(batch)).cpu().numpy()
                    else:
                        scores = np.concatenate([expit(session.run(None, {input_name: row[None]})[0]) for row in inputs])
                    probabilities[start:start + len(selected)] = scores
                    if start % (args.batch_size * 40) == 0:
                        print(f'{stem}: {start + len(selected):,}/{len(indices):,}', flush=True)
                if not np.isfinite(probabilities).all():
                    raise RuntimeError('Nonfinite probabilities')
                np.savez_compressed(output, record_ids=record_ids, targets=targets,
                                    positive_only_mask=primary, conditional_mask=conditional,
                                    probabilities=probabilities, thresholds=predictor.thresholds)
                save_json(meta_path, metadata)
            if digest(artifact) != artifact_hash or digest(path) != before:
                raise RuntimeError('Model changed during evaluation')
            row = {'name': stem, 'checkpoint': str(path), 'artifact': str(artifact),
                   **metadata, 'normalization': predictor.filter_config.normalization,
                   'thresholds': predictor.thresholds.tolist(),
                   'positive_only_metrics': summarize_metrics(targets, primary, probabilities, predictor.thresholds, mode='positive_only'),
                   'conditional_metrics': summarize_metrics(targets, conditional, probabilities, predictor.thresholds, mode='conditional_absence_negative')}
            results.append(row)
            model_hashes[str(path)] = before
            save_json(OUT / 'partial_results.json', results)
            print(stem, 'positive Recall', row['positive_only_metrics']['macro_recall'],
                  'conditional F1', row['conditional_metrics']['macro_f1'], flush=True)
    summary = {'dataset': 'Chapman-Shaoxing-Ningbo 1.0.0', 'mapping_version': MAPPING_VERSION,
               'headers_scanned': len(headers), 'waveforms_evaluated': len(indices),
               'waveforms_excluded': len(rows) - len(indices), 'waveform_errors': errors,
               'checksum_verification_stamp': stamp.read_text().strip(),
               'dataset_sha256_manifest': digest(dataset / 'SHA256SUMS.txt'),
               'official_dictionary_sha256': {str(p): digest(p) for p in dictionary_files},
               'preprocessing_parity': preprocess_checks,
               'no_external_training_or_threshold_tuning': True,
               'model_selection_on_external_data': False, 'negative_truth_clinically_verified': False,
               'model_hashes_unchanged': model_hashes, 'results': results}
    save_json(OUT / 'summary.json', summary)
    write_report(summary)
    print('Complete; report outputs/reports/Chapman_冻结模型外部验证总结.md', flush=True)


if __name__ == '__main__':
    main()
