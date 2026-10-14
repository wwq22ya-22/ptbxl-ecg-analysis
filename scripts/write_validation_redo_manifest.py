"""Package already-frozen fold9 thresholds without choosing or evaluating."""
from pathlib import Path
import json,hashlib,copy
ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'src').exists():ROOT=Path('D:/MyProject/college/ecg1')
OUT=ROOT/'outputs/validation_only_redo_20261010'
frozen=json.loads((ROOT/'outputs/further_optimization/final_selection.json').read_text(encoding='utf-8'))
result=json.loads((OUT/'joint_thresholds/result.json').read_text(encoding='utf-8'))
assert result['solver']['optimal'] and result['cv_model_choice']=='ensemble_equal'
for member in frozen['components']:
    with Path(member['checkpoint']).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==member['sha256']
manifest=copy.deepcopy(frozen)
manifest.update(thresholds=result['thresholds'],validation_metrics=result['validation'],fold10_evaluation=None,description='2026-10-10纠正：复用原训练所得50/50模型，仅按fold9重求联合约束阈值；未评估fold10。')
manifest['threshold_policy'].update(feasible=True,method='fold9_only_joint_PR_MILP_redo_20261010',solver=result['solver'])
target=OUT/'joint_thresholds/ensemble_manifest.json'
target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Packaged frozen development-only ensemble manifest:',target)
