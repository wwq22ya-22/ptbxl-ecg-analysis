"""Keep replay cohorts and logged model contracts aligned across GUI/CLI paths."""
import json
import numpy as np
import pytest
from src.inference import serial_samples as module
from scripts.ecg_stm32_viz import transmission_sample

@pytest.fixture
def bundle(tmp_path, monkeypatch):
    manifest = {'protocol': 1, 'model_sha256': 'a' * 64,
                'labels': ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
                'lead_order': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
                'input_shape': [12, 1000], 'thresholds': [.55, .60, .55, .75, .65],
                'input_scale': .2, 'input_zero_point': 0,
                'preprocessing': {'normalization': 'train_global', 'training_mean': [0.] * 12},
                'preprocessing_location': 'host'}
    arrays = {'raw_mv': np.ones((2, 12, 1000), np.float32), 'ecg': np.zeros((2, 12, 1000), np.float32),
              'targets': np.array([[1, 0, 0, 0, 0], [0, 1, 0, 1, 0]], np.uint8),
              'ecg_id': np.array([47, 99]), 'strat_fold': np.array([10, 10]),
              'reference_probabilities': np.ones((2, 5), np.float32) * .4}
    source = {key: manifest[key] for key in module.CONTRACT_KEYS}
    source.update(fold=10, sample_file='samples.npz', sample_count=2,
                  sample_source='PTB-XL historical test fold 10', require_diagnostic_label=True,
                  reference='Cube.AI host C-model; not physical-board outputs')
    def write():
        np.savez(tmp_path / 'samples.npz', **arrays)
        source['sample_sha256'] = module.digest(tmp_path / 'samples.npz')
        (tmp_path / 'sample_manifest.json').write_text(json.dumps(source), encoding='utf-8')
    write()
    monkeypatch.setattr(module, 'FOLD10_DIRECTORY', tmp_path)
    return manifest, source, arrays, write, tmp_path

def test_fold10_supports_both_pipelines_and_logs_source(bundle):
    manifest, source, _, _, directory = bundle
    host = module.read_samples(manifest, directory, 10)
    board_manifest = dict(manifest, protocol=2, preprocessing_location='board', preprocessing_sha256='b' * 64)
    board = module.read_samples(board_manifest, directory, 10)
    np.testing.assert_array_equal(transmission_sample(manifest, host, 1), 0)
    np.testing.assert_array_equal(transmission_sample(board_manifest, board, 1), 1)
    metadata = module.result_metadata(board_manifest, board, 1)
    assert metadata['fold'] == 10 and metadata['ecg_id'] == 99
    assert metadata['truth'] == ['MI', 'CD']
    assert metadata['sample_sha256'] == source['sample_sha256']
    assert metadata['preprocessing_location'] == 'board' and metadata['protocol'] == 2

@pytest.mark.parametrize('key,value', [('model_sha256', 'b' * 64), ('thresholds', [.1] * 5),
                                      ('input_scale', .4), ('preprocessing', {'normalization': 'zscore'})])
def test_fold10_wrong_contract_rejected(bundle, key, value):
    manifest, source, _, write, directory = bundle
    source[key] = value; write()
    with pytest.raises(ValueError, match='不一致'):
        module.read_samples(manifest, directory, 10)

@pytest.mark.parametrize('kind', ['mixed_fold', 'zero_labels', 'invalid_probabilities', 'duplicate_ids'])
def test_corrupt_cohort_rejected(bundle, kind):
    manifest, _, arrays, write, directory = bundle
    if kind == 'mixed_fold': arrays['strat_fold'][1] = 9
    if kind == 'zero_labels': arrays['targets'][1] = 0
    if kind == 'invalid_probabilities': arrays['reference_probabilities'][1, 0] = np.nan
    if kind == 'duplicate_ids': arrays['ecg_id'][1] = arrays['ecg_id'][0]
    write()
    with pytest.raises(ValueError): module.read_samples(manifest, directory, 10)

def test_fold9_legacy_bundle_still_supported_and_labeled(bundle):
    manifest, source, _, _, directory = bundle
    manifest.update(sample_file=source['sample_file'], sample_count=2,
                    sample_sha256=source['sample_sha256'], sample_source='PTB-XL validation fold 9')
    samples = module.read_samples(manifest, directory)
    assert module.result_metadata(manifest, samples, 0)['fold'] == 9
    assert samples['_metadata']['source_label'] == '验证折9（联调）'

def test_changed_sample_bytes_rejected(bundle):
    manifest, source, _, _, directory = bundle
    (directory / source['sample_file']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='波形包已变化'):
        module.read_samples(manifest, directory, 10)
