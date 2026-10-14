"""Exploratory follow-up: three-seed teachers; validation selection before test.

This follow-up was conceived after reviewing the parent study's test results,
so it is explicitly exploratory on a previously reused test fold, not a new
blind or preregistered test. Equal weights, all three seeds, no test tuning.
"""
from __future__ import annotations
import copy
import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest, get_split
from scripts.summarize_teacher_study import section
from src.data.ptbxl import SUPERCLASSES
from src.models.probability_ensemble import load_teacher
from src.training.metrics import calculate_multilabel_metrics, optimize_thresholds
from src.training.train import set_seed, _save_checkpoint

PARENT = ROOT / "outputs/teacher_preprocessing"
OUT = PARENT / "ensemble_followup"
KEYS = ("macro_auroc", "macro_auprc", "macro_precision", "macro_recall", "macro_f1")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".saving")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric(truth, scores, thresholds):
    value = calculate_multilabel_metrics(truth, scores, thresholds)
    ap = average_precision_score(truth, scores, average=None)
    value["macro_auprc"] = float(np.mean(ap))
    for i, label in enumerate(SUPERCLASSES):
        value["per_class"][label]["auprc"] = float(ap[i])
    return value


def predict(path, split, *, return_logits=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cp = load_teacher(path, device)
    values, frame = get_split(split, cp)
    scores = []
    with torch.inference_mode():
        for start in range(0, len(frame), 32):
            inputs = torch.from_numpy(np.array(values[start:start+32], copy=True)).to(device)
            logits = model(inputs)
            scores.append((logits if return_logits else logits.sigmoid()).cpu().numpy())
            if return_logits and start % 4096 == 0:
                print(f"Aligned training logits: {min(start+32,len(frame))}/{len(frame)}", flush=True)
    del model
    return np.concatenate(scores), frame, cp


def bundle(parent_rows, model_type):
    selected = sorted([r for r in parent_rows if r["model"] == model_type and r["preprocessing"] == "P0_current"],
                      key=lambda r: r["seed"])
    assert [r["seed"] for r in selected] == [42, 43, 44]
    members = []
    configs = []
    for r in selected:
        cp = torch.load(r["checkpoint"], map_location="cpu", weights_only=True)
        assert digest(Path(r["checkpoint"])) == r["checkpoint_sha256"]
        configs.append(cp["config"])
        members.append({"seed": r["seed"], "model_type": cp["model_type"],
                        "dropout": cp["config"]["model"]["dropout"], "model_state": cp["model_state"],
                        "checkpoint_sha256": r["checkpoint_sha256"]})
    assert all(c["signal_processing"] == configs[0]["signal_processing"] for c in configs)
    template = torch.load(selected[0]["checkpoint"], map_location="cpu", weights_only=True)
    cfg = copy.deepcopy(template["config"])
    cfg["model"]["type"] = "probability_ensemble"
    return {"model_type": "probability_ensemble", "members": members, "config": cfg,
            "thresholds": [.5] * 5, "sampling_rate_hz": 100, "superclasses": list(SUPERCLASSES),
            "lead_names": template["lead_names"], "training_records": 17084, "validation_records": 2146,
            "member_weights": [1/3] * 3, "teacher_family": model_type, "seeds": [42,43,44],
            "purpose": "PC-side frozen multi-label teacher; no board firmware replacement"}


def evaluate(path, split, name):
    scores, frame, cp = predict(path, split)
    truth = frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    thresholds = optimize_thresholds(truth, scores) if split == "validation" else cp["thresholds"]
    value = metric(truth, scores, thresholds)
    if split == "validation":
        cp["thresholds"] = thresholds
        cp["validation_metrics_tuned"] = value
        _save_checkpoint(path, cp)
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / f"{split}_predictions.npz", ecg_ids=frame.ecg_id.to_numpy(),
                        patient_ids=frame.patient_id.to_numpy(), targets=truth, probabilities=scores,
                        thresholds=np.asarray(thresholds))
    save(folder / f"{split}.json", value)
    print(f"{name} {split}: P={value['macro_precision']:.6f} R={value['macro_recall']:.6f} F1={value['macro_f1']:.6f}", flush=True)
    return value


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    set_seed(42)
    parent = read(PARENT / "summary.json")
    protected = parent["protocol"]["protected_artifacts"]
    source_files = [Path(__file__), ROOT / "src/models/probability_ensemble.py"]
    policy = {"phase": "exploratory_three_seed_ensemble_followup", "proposed_after_parent_test_review": True,
              "test_is_new_blind_test": False, "candidates": ["inception1d", "inception_time1d"],
              "seeds": [42,43,44], "weights": "equal probability average; no weight tuning",
              "preprocessing": "P0_current; identical training-only per-lead normalization",
              "selection": "validation macro F1 then AUROC, include parent frozen single-teacher candidate",
              "thresholds": "original coarse grid using validation only", "test_used_for_this_selection": False,
              "source_sha256": {str(p.relative_to(ROOT)):digest(p) for p in source_files},
              "protected_artifacts": protected}
    if (OUT / "protocol.json").exists():
        assert read(OUT / "protocol.json") == policy
        if (OUT / "summary.json").exists():
            print("Follow-up already complete", flush=True)
            return
        raise RuntimeError("Partial follow-up: inspect existing validation/test outputs before restarting.")
    save(OUT / "protocol.json", policy)
    rows = []
    for family in policy["candidates"]:
        name = family + "_ensemble3"
        path = OUT / f"{name}.pt"
        _save_checkpoint(path, bundle(parent["runs"], family))
        validation = evaluate(path, "validation", name)
        rows.append({"name":name,"checkpoint":str(path),"checkpoint_sha256":digest(path),"validation":validation})
    original = parent["selected_teacher"]
    candidates = rows + [{"name":original["name"],"checkpoint":original["checkpoint"],
                          "checkpoint_sha256":original["checkpoint_sha256"],"validation":original["validation"]}]
    selected = max(candidates, key=lambda r:(r["validation"]["macro_f1"],r["validation"]["macro_auroc"]))
    save(OUT / "frozen_selection.json", {"candidates": candidates, "selected":selected,
                                         "frozen_at":datetime.datetime.now().isoformat(),"test_used":False})
    shutil.copyfile(selected["checkpoint"], OUT / "recommended_teacher.pt")
    print(f"Validation choice frozen: {selected['name']}; evaluating test next", flush=True)
    for row in rows:
        row["test"] = evaluate(Path(row["checkpoint"]), "test", row["name"])
    if selected["name"] == original["name"]:
        selected["test"] = original["test"]
    # Independent count-based check, separate from the metric helper.
    for row in rows:
        for split in ("validation","test"):
            with np.load(OUT / row["name"] / f"{split}_predictions.npz") as a:
                t = a["targets"].astype(bool); p = a["probabilities"] >= a["thresholds"]
                truth = a["targets"]
                for i,label in enumerate(SUPERCLASSES):
                    tp=int((t[:,i]&p[:,i]).sum()); fp=int((~t[:,i]&p[:,i]).sum())
                    fn=int((t[:,i]&~p[:,i]).sum()); tn=int((~t[:,i]&~p[:,i]).sum())
                    c=row[split]["per_class"][label]
                    assert c["confusion_matrix"] == [[tn,fp],[fn,tp]]
                    for key, actual in dict(precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                          f1=2*tp/max(1,2*tp+fp+fn),auroc=roc_auc_score(truth[:,i],a['probabilities'][:,i]),
                          auprc=average_precision_score(truth[:,i],a['probabilities'][:,i])).items():
                        assert abs(c[key]-actual)<1e-12
    logits, train_frame, cp = predict(OUT / "recommended_teacher.pt", "train", return_logits=True)
    assert logits.shape==(17084,5) and np.isfinite(logits).all()
    sha = digest(OUT / "recommended_teacher.pt")
    np.savez_compressed(OUT / "teacher_training_logits.npz", ecg_ids=train_frame.ecg_id.to_numpy(),
                        logits=logits, teacher_sha256=sha, superclasses=np.asarray(SUPERCLASSES))
    for relative, expected in protected.items():
        assert digest(ROOT / relative)==expected
    save(OUT / "audit.json", {"passed":True,"prediction_sets_recomputed":4,
                              "protected_artifacts_unchanged":True,"training_logits_records":len(logits),
                              "teacher_sha256":sha,"selection_frozen_before_followup_test":True})
    save(OUT / "summary.json", {"protocol":policy,"runs":rows,"selected_teacher":selected,
                                "training_logits":str(OUT / 'teacher_training_logits.npz'),"teacher_sha256":sha})
    lines = ["### 探索性追加：三种子教师集成", "",
             "在查看首轮单模型测试结果后追加，明确属于探索性实验，不能称预注册或新的盲测。没有新增训练；将42/43/44的独立概率等权平均。两种架构都构建集成作对照，阈值和候选仍只按验证集选择，并在追加测试前冻结。",
             "", "| 教师 | 验证F1 | 测试AUROC | 测试AUPRC | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['validation']['macro_f1']:.6f} | "+" | ".join(f"{row['test'][k]:.6f}" for k in KEYS)+" |")
    lines += ["",f"按验证集选出的追加研究教师：{selected['name']}。可直接用于后续蒸馏的冻结教师权重位于 `outputs/teacher_preprocessing/ensemble_followup/recommended_teacher.pt`，17084条训练记录对齐的教师logits位于同目录 `teacher_training_logits.npz`。",
              "训练logits仅来自训练患者，没有混入验证或测试记录。集成是电脑端教师，当前seed44 INT8固件保持原样；新学生的精度和资源仍需单独蒸馏、量化验证。",
              "4份预测的逐类指标、混淆矩阵独立复算通过；原模型/固件SHA256检查通过。复现：`.venv/Scripts/python.exe scripts/compose_teacher_ensembles.py`。"]
    section(ROOT / "md/模型训练与优化.md", "\n".join(lines), "teacher-ensemble-followup")
    details=["## 探索性追加：三种子教师集成", "", "| 教师 | 类别 | 阳性数 | AUROC | AUPRC | Precision | Recall | F1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        for label in SUPERCLASSES:
            c=row['test']['per_class'][label]
            details.append(f"| {row['name']} | {label} | {c['support']} | "+" | ".join(f"{c[k]:.6f}" for k in ('auroc','auprc','precision','recall','f1'))+" |")
    section(ROOT / "md/各模型整体与逐类指标总表.md", "\n".join(details), "teacher-ensemble-followup")
    entry=f"教师探索性追加实验：三种子概率集成的验证选择为 `{selected['name']}`；权重和训练logits位于 `outputs/teacher_preprocessing/ensemble_followup/`。该步骤在首轮测试分析后提出，属于固定测试集上的探索性复核；当前板上候选继续使用原seed44 INT8。结果见现有训练文档和指标总表。"
    for path in (ROOT / "README.md",ROOT / "md/README.md"):
        section(path,entry,"teacher-ensemble-followup")
    print("Follow-up complete: two ensembles, validation selection, all-class audit, training-only KD logits, existing MDs updated", flush=True)


if __name__ == "__main__":
    main()
