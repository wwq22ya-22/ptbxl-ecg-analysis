"""Load fold-specific replay samples without changing the firmware contract."""
import json
from pathlib import Path
import numpy as np
from scripts.prepare_stm32 import digest

ROOT = Path(__file__).resolve().parents[2]
FOLD10_DIRECTORY = ROOT / 'outputs/serial_samples/seed44_fold10'
CONTRACT_KEYS = ('model_sha256', 'labels', 'lead_order', 'input_shape', 'thresholds',
                 'input_scale', 'input_zero_point', 'preprocessing')
FOLD_CHOICES = {'折9（联调）': 9, '折10（历史测试）': 10}

def read_samples(manifest, directory, fold=9):
    if fold not in (9, 10):
        raise ValueError('串口回放仅支持折9或折10。')
    if fold == 10:
        manifest_path = FOLD10_DIRECTORY / 'sample_manifest.json'
        if not manifest_path.is_file():
            raise ValueError('缺少折10波形包，请先运行 scripts/prepare_serial_fold10.py。')
        source = json.loads(manifest_path.read_text(encoding='utf-8'))
        if source.get('fold') != 10 or any(source.get(key) != manifest.get(key) for key in CONTRACT_KEYS):
            raise ValueError('折10波形包与当前模型、预处理、量化参数或阈值不一致。')
        directory = FOLD10_DIRECTORY
    else:
        source = manifest
    path = Path(directory) / source['sample_file']
    if digest(path) != source['sample_sha256']:
        raise ValueError('验证波形包已变化，请重新生成配套样本包。')
    with np.load(path, allow_pickle=False) as data:
        samples = {name: data[name].copy() for name in data.files}
    n = len(samples['ecg_id'])
    if n < 1 or n != source['sample_count'] or len(np.unique(samples['ecg_id'])) != n:
        raise ValueError('样本数量或记录ID无效。')
    for key in ('ecg', 'raw_mv'):
        if samples[key].shape != (n, 12, 1000) or not np.isfinite(samples[key]).all():
            raise ValueError(f'波形 {key} 形状或数值不正确。')
    if samples['targets'].shape != (n, 5) or not np.isin(samples['targets'], [0, 1]).all():
        raise ValueError('标签数组无效。')
    probabilities = samples['reference_probabilities']
    if (probabilities.shape != (n, 5) or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError('电脑参考概率无效。')
    if fold == 10:
        if ('strat_fold' not in samples or samples['strat_fold'].shape != (n,)
                or not np.all(samples['strat_fold'] == 10)):
            raise ValueError('折10波形包混入其他划分。')
        if source.get('require_diagnostic_label') and np.any(samples['targets'].sum(axis=1) == 0):
            raise ValueError('筛选后的折10波形包包含五类全零记录。')
    samples['_metadata'] = {'fold': fold, 'sample_count': n,
                            'sample_source': source['sample_source'],
                            'sample_sha256': source['sample_sha256'],
                            'reference': source.get('reference', 'Cube.AI host C-model'),
                            'source_label': '验证折9（联调）' if fold == 9 else '折10（历史测试）'}
    return samples

def result_metadata(manifest, samples, index):
    metadata = samples['_metadata']
    return {'ecg_id': int(samples['ecg_id'][index]), 'sample_index': index,
            'fold': metadata['fold'], 'sample_source': metadata['sample_source'],
            'sample_sha256': metadata['sample_sha256'], 'model_sha256': manifest['model_sha256'],
            'protocol': manifest['protocol'],
            'preprocessing_location': manifest.get('preprocessing_location', 'host'),
            'preprocessing_sha256': manifest.get('preprocessing_sha256'),
            'thresholds': list(manifest['thresholds']),
            'truth': [name for name, hit in zip(manifest['labels'], samples['targets'][index]) if hit],
            'host_c_reference_probabilities': samples['reference_probabilities'][index].tolist(),
            'reference': metadata['reference']}
