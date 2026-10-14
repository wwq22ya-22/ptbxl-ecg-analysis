"""Prespecified patient-safe preprocessing ablation and larger teacher study.

No test prediction is made until all validation selections are frozen. Existing
deployed students and firmware are protected by checksums. Resume completed
runs, but reject interrupted training instead of silently calling it complete.
"""
from __future__ import annotations

import copy
import datetime
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest
from src.data.ptbxl import SUPERCLASSES
from src.models import build_model
from src.signal_processing.filters import FilterConfig
from src.training.data import ensure_preprocessed_cache, fit_training_normalization, load_labeled_splits
from src.training.metrics import calculate_multilabel_metrics, optimize_thresholds
from src.training.train import train_baseline

OUT = ROOT / "outputs/teacher_preprocessing"
BASE = ROOT / "outputs/distillation/distilled_seed44/checkpoint/cnn1d_best.pt"
SEEDS = (42, 43, 44)
PREPROCESSING = {
    "P0_current": {"baseline_removal": True, "lowcut_hz": 0.5},
    "P1_bandpass_only": {"baseline_removal": False, "lowcut_hz": 0.5},
    "P2_gentle_bandpass": {"baseline_removal": False, "lowcut_hz": 0.1},
}
STRONG_MODELS = ("se_resnet18", "inception_time1d")
KEYS = ("macro_auroc", "macro_auprc", "macro_precision", "macro_recall", "macro_f1")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".saving")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def status(stage, **extra):
    print(stage, flush=True)
    save(OUT / "status.json", {"stage": stage, "updated_at": datetime.datetime.now().isoformat(), **extra})


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def run_name(model, preprocessing, seed):
    return f"{model}_{preprocessing}_seed{seed}"


def validation_key(row):
    return row["validation"]["macro_f1"], row["validation"]["macro_auroc"]


def metrics(targets, probabilities, thresholds):
    value = calculate_multilabel_metrics(targets, probabilities, thresholds)
    ap = average_precision_score(targets, probabilities, average=None)
    value["macro_auprc"] = float(np.mean(ap))
    for index, label in enumerate(SUPERCLASSES):
        value["per_class"][label]["auprc"] = float(ap[index])
    return value


def evaluate(path, frame, split, result_dir):
    report = result_dir / f"{split}.json"
    predictions = result_dir / f"{split}_predictions.npz"
    checkpoint = load(path)
    cfg = checkpoint["config"]
    if report.exists() and predictions.exists():
        record = read(report)
        assert record["checkpoint_sha256"] == digest(path)
        with np.load(predictions) as arrays:
            np.testing.assert_array_equal(arrays["ecg_ids"], frame.ecg_id)
        return record["metrics"]
    assert not report.exists() and not predictions.exists(), "Incomplete evaluation; inspect before retrying."
    cache = ensure_preprocessed_cache(
        frame, cfg["dataset"]["root"], ROOT / cfg["outputs"]["cache_dir"],
        split_name=f"{split}_full", sampling_rate_hz=100,
        config=FilterConfig.from_mapping(cfg["signal_processing"]),
    )
    x = np.load(cache, mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint["model_type"], dropout=cfg["model"]["dropout"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(frame), 32):
            inputs = torch.from_numpy(np.array(x[start:start+32], copy=True)).to(device)
            probabilities.append(torch.sigmoid(model(inputs)).cpu().numpy())
    scores = np.concatenate(probabilities)
    targets = frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    thresholds = optimize_thresholds(targets, scores) if split == "validation" else checkpoint["thresholds"]
    value = metrics(targets, scores, thresholds)
    if split == "validation":
        # Recalibrate using deterministic FP32 validation inference, never test.
        checkpoint["thresholds"] = thresholds
        checkpoint["validation_metrics_tuned"] = value
        checkpoint["validation_inference_precision"] = "FP32"
        from src.training.train import _save_checkpoint
        _save_checkpoint(path, checkpoint)
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions, ecg_ids=frame.ecg_id.to_numpy(), patient_ids=frame.patient_id.to_numpy(),
                        targets=targets, probabilities=scores, thresholds=np.asarray(thresholds))
    save(report, {"metrics": value, "checkpoint_sha256": digest(path), "cache": str(cache),
                  "inference": "FP32; thresholds fitted on validation only"})
    del model
    return value


def trial_config(base, model, preprocessing, seed):
    cfg = copy.deepcopy(base)
    cfg["dataset"]["root"] = "D:/datasets/ptb-xl"
    cfg["dataset"]["require_diagnostic_label"] = True
    cfg["signal_processing"].update(PREPROCESSING[preprocessing])
    cfg["signal_processing"].update(normalization="train_global", highcut_hz=40.0)
    for key in ("training_mean", "training_std"):
        cfg["signal_processing"].pop(key, None)
    cfg["model"].update(type=model, dropout=0.2)
    cfg["training"] = {
        "epochs": 30, "batch_size": 32, "learning_rate": 0.001, "weight_decay": 0.0001,
        "early_stopping_patience": 8, "num_workers": 0, "use_amp": True,
        "random_seed": seed, "loss": "bce", "pos_weight_power": 1.0, "scheduler": "none",
    }
    folder = OUT / run_name(model, preprocessing, seed)
    for key, sub in (("checkpoint_dir", "checkpoint"), ("metrics_dir", "metrics"),
                     ("figure_dir", "figure"), ("predictions_dir", "predictions")):
        cfg["outputs"][key] = str((folder / sub).relative_to(ROOT))
    cfg["outputs"]["cache_dir"] = "outputs/cache"
    return cfg


def train_one(base, frames, model, preprocessing, seed):
    name = run_name(model, preprocessing, seed)
    folder = OUT / name
    checkpoint = folder / f"checkpoint/{model}_best.pt"
    history = folder / f"metrics/training_history_{model}_full.json"
    cfg = trial_config(base, model, preprocessing, seed)
    if (folder / "config.json").exists():
        assert read(folder / "config.json") == cfg, "Existing training config changed."
    else:
        save(folder / "config.json", cfg)
    if not history.exists():
        assert not checkpoint.exists(), f"Interrupted training in {name}; inspect and resume explicitly."
        status(f"Training {name}: maximum 30 epochs; validation-only AUROC checkpoint selection")
        train_baseline(cfg, project_root=ROOT, model_type=model)
    cp = load(checkpoint)
    assert cp["training_records"] == 17084 and cp["validation_records"] == 2146
    validation = evaluate(checkpoint, frames["validation"], "validation", folder / "evaluation")
    parameter_count = sum(p.numel() for p in build_model(model).parameters())
    row = {"name": name, "model": model, "preprocessing": preprocessing, "seed": seed,
           "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
           "parameters": parameter_count, "epoch": cp["epoch"], "validation": validation,
           "epochs_completed": len(read(history)["history"])}
    save(folder / "result.json", row)
    status(f"Completed {name}: validation F1 {validation['macro_f1']:.6f}, "
           f"P {validation['macro_precision']:.6f}, R {validation['macro_recall']:.6f}")
    return row


def replace_section(path, text):
    start, end = "<!-- teacher-preprocessing:start -->", "<!-- teacher-preprocessing:end -->"
    previous = path.read_text(encoding="utf-8")
    section = start + "\n" + text + "\n" + end
    if start in previous:
        left, remaining = previous.split(start, 1)
        _, right = remaining.split(end, 1)
        updated = left + section + right
    else:
        updated = previous.rstrip() + "\n\n" + section + "\n"
    path.write_text(updated, encoding="utf-8")


def report(summary=None):
    intro = ["## 2026-10-07：预处理消融与更强教师", "",
             "本轮只比较预处理与教师架构，不同时加入新增强、蒸馏或新的训练损失。仍为十二导联、100 Hz、10 秒、五类多标签。",
             "剔除无五类标签记录：训练 17084、验证 2146、测试 2158；使用官方患者隔离 folds 1–8 / 9 / 10。",
             "", "| 预处理 | 中值基线去除 | 带通 |", "|---|---|---|",
             "| P0_current | 0.2/0.6 秒，保留现有处理 | 0.5–40 Hz |",
             "| P1_bandpass_only | 关闭 | 0.5–40 Hz |",
             "| P2_gentle_bandpass | 关闭 | 0.1–40 Hz |", "",
             "每个预处理方案仅用对应训练记录重新拟合逐导联均值和标准差。P0→P1 仅去掉中值基线处理，P1→P2 仅改变低频截止。",
             "固定 batch32、AdamW lr0.001、weight decay0.0001、BCE正类权重、dropout0.2、最多30轮、早停8轮；按验证AUROC选轮次，FP32验证推理后用原粗网格选择逐类F1阈值。",
             "先以 seed42 的紧凑 Inception 比较三种预处理并按验证F1/AUROC选择一种；再于原P0与入选预处理上比较两个更大教师。",
             "更大教师：SE-ResNet18（32/64/128/256通道，8个SE残差块、平均和最大池化）；InceptionTime单网络（6模块、每支32通道、40/20/10卷积核、每3模块残差、保留时间分辨率）。这是项目PyTorch实现，不宣称复现论文分数或原5网络集成。",
             "更大教师家族按seed42验证F1/AUROC选定后，用43/44复核；紧凑Inception的P0和入选预处理也补43/44，作配对对照。最终教师种子只按验证指标选择，所有测试预测在冻结选择后生成。",
             "方法依据：[InceptionTime作者实现](https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py)、[SENet作者仓库](https://github.com/hujie-frank/SENet)。",
             "现有学生INT8、旧教师、固件以SHA256保护；不替换当前部署。fold10此前多次使用，结果属于同一固定测试集复核，不能称新盲测。", ""]
    if summary is None:
        intro += ["状态：方案已预先记录，实验进行中。配置、权重、预测与阶段记录保存于 `outputs/teacher_preprocessing/`。"]
        replace_section(ROOT / "md/模型训练与优化.md", "\n".join(intro))
        return
    rows = summary["runs"]
    intro += [f"入选预处理（seed42筛选）：{summary['selection']['preprocessing']}；复核教师家族：{summary['selection']['teacher_model']} / {summary['selection']['teacher_preprocessing']}。",
              "", "| 模型 | 预处理 | 种子 | 参数 | 最佳轮次 | 验证F1 | 测试AUROC | 测试AUPRC | Precision | Recall | F1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        intro.append(f"| {r['model']} | {r['preprocessing']} | {r['seed']} | {r['parameters']} | {r['epoch']} | {r['validation']['macro_f1']:.6f} | "
                     + " | ".join(f"{r['test'][key]:.6f}" for key in KEYS) + " |")
    intro += ["", "| 三种子对照 | 测试F1均值 ± 样本标准差 | Precision均值 | Recall均值 | HYP Recall均值 |",
              "|---|---:|---:|---:|---:|"]
    for family, stats in summary["seed_statistics"].items():
        intro.append(f"| {family} | {stats['macro_f1']['mean']:.6f} ± {stats['macro_f1']['std']:.6f} | "
                     f"{stats['macro_precision']['mean']:.6f} | {stats['macro_recall']['mean']:.6f} | {stats['hyp_recall']['mean']:.6f} |")
    chosen = summary["selected_teacher"]
    intro += ["", f"按验证指标冻结的教师候选：{chosen['name']}，已另存 `outputs/teacher_preprocessing/selected_teacher.pt`。较大网络是电脑端教师；尚未蒸馏、量化或完成板上资源分析。",
              "与当前seed44 INT8的比较只是部署参考；架构、精度格式不同，不能把教师得分等同于新学生可达到的得分。预处理/架构效应应看本轮同训练预算、同种子的配对结果。",
              "", "复现：`.venv/Scripts/python.exe scripts/run_teacher_preprocessing.py`。结果：`summary.json`；测试前选择：`frozen_selection.json`；独立预测复算：`audit.json`。"]
    replace_section(ROOT / "md/模型训练与优化.md", "\n".join(intro))
    detail = ["## 2026-10-07：预处理消融与更强教师", "",
              "同一筛选后患者隔离测试集2158条；以下均为FP32教师，各类阈值仅用验证集确定。所有候选都报告，未按测试分数选模型。",
              "", "| 模型 | 预处理 | 种子 | 类别 | 阳性数 | AUROC | AUPRC | Precision | Recall | F1 |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        for label in SUPERCLASSES:
            c = r["test"]["per_class"][label]
            detail.append(f"| {r['model']} | {r['preprocessing']} | {r['seed']} | {label} | {c['support']} | "
                          + " | ".join(f"{c[k]:.6f}" for k in ("auroc", "auprc", "precision", "recall", "f1")) + " |")
    replace_section(ROOT / "md/各模型整体与逐类指标总表.md", "\n".join(detail))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    base = load(BASE)["config"]
    frames = load_labeled_splits("D:/datasets/ptb-xl", 100, require_diagnostic_label=True)
    expected = {"train": 17084, "validation": 2146, "test": 2158}
    assert {s: len(f) for s, f in frames.items()} == expected
    patients = {s: set(f.patient_id) for s, f in frames.items()}
    assert not (patients["train"] & patients["validation"] or patients["train"] & patients["test"]
                or patients["validation"] & patients["test"])
    assert all((f[list(SUPERCLASSES)].sum(axis=1) > 0).all() for f in frames.values())
    protected = [BASE, ROOT / "outputs/optimization/optimized_best.pt",
                 ROOT / "outputs/distillation/distilled_export/ptbxl_cnn1d_int8.onnx",
                 ROOT / "outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_STM32F407VE.hex"]
    protocol = {"experiment": "teacher_preprocessing_20261007", "cohort": expected,
                "preprocessing": PREPROCESSING, "teacher_models": STRONG_MODELS,
                "screen_seed": 42, "replication_seeds": SEEDS,
                "training": trial_config(base, "inception1d", "P0_current", 42)["training"],
                "checkpoint_selection": "validation AUROC", "thresholds": "original coarse grid, FP32 validation F1",
                "selection": "validation macro F1; validation AUROC tie break",
                "test_used_for_selection": False, "test_previously_reused": True,
                "protected_artifacts": {str(p.relative_to(ROOT)): digest(p) for p in protected},
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                                  [Path(__file__), ROOT / "src/models/teachers.py", ROOT / "src/training/train.py"]}}
    # Tuples must compare in their JSON-compatible representation on resume.
    protocol = json.loads(json.dumps(protocol))
    if (OUT / "protocol.json").exists():
        assert read(OUT / "protocol.json") == protocol, "Protocol changed; use a new experiment."
    else:
        save(OUT / "protocol.json", protocol)
        for split, frame in frames.items():
            frame[["ecg_id", "patient_id", "strat_fold", *SUPERCLASSES]].to_csv(OUT / f"cohort_{split}.csv", index=False)
    if (OUT / "summary.json").exists():
        report(read(OUT / "summary.json"))
        return
    report()
    runs = {}

    def run(model, preprocessing, seed):
        name = run_name(model, preprocessing, seed)
        if name not in runs:
            runs[name] = train_one(base, frames, model, preprocessing, seed)
        return runs[name]

    preprocessing_trials = [run("inception1d", p, 42) for p in PREPROCESSING]
    selected_preprocessing = max(preprocessing_trials, key=validation_key)["preprocessing"]
    save(OUT / "preprocessing_selection.json", {"selected": selected_preprocessing,
                                               "validation_trials": preprocessing_trials, "test_used": False})
    strong_trials = [run(model, p, 42) for p in dict.fromkeys(("P0_current", selected_preprocessing)) for model in STRONG_MODELS]
    family = max(strong_trials, key=validation_key)
    selection = {"preprocessing": selected_preprocessing, "teacher_model": family["model"],
                 "teacher_preprocessing": family["preprocessing"], "screen_seed": 42, "test_used": False}
    save(OUT / "family_selection.json", selection)
    for seed in (43, 44):
        for p in dict.fromkeys(("P0_current", selected_preprocessing, family["preprocessing"])):
            run("inception1d", p, seed)
        run(family["model"], family["preprocessing"], seed)
    finalists = [r for r in runs.values() if r["model"] == family["model"] and r["preprocessing"] == family["preprocessing"]]
    selected = max(finalists, key=validation_key)
    frozen = {"policy": selection, "selected_teacher": selected, "all_validation_results": list(runs.values()),
              "test_used": False, "frozen_at": datetime.datetime.now().isoformat()}
    save(OUT / "frozen_selection.json", frozen)
    shutil.copyfile(selected["checkpoint"], OUT / "selected_teacher.pt")
    status("All selections frozen; evaluating the fixed fold-10 test cohort")
    for row in runs.values():
        row["test"] = evaluate(Path(row["checkpoint"]), frames["test"], "test", OUT / row["name"] / "evaluation")
        save(OUT / row["name"] / "result.json", row)
    stats = {}
    for model, p in dict.fromkeys((r["model"], r["preprocessing"]) for r in runs.values()):
        members = [r for r in runs.values() if r["model"] == model and r["preprocessing"] == p]
        if len(members) != 3:
            continue
        values = {key: [r["test"][key] for r in sorted(members, key=lambda r: r["seed"])] for key in KEYS}
        values["hyp_recall"] = [r["test"]["per_class"]["HYP"]["recall"] for r in sorted(members, key=lambda r: r["seed"])]
        stats[f"{model}/{p}"] = {key: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v}
                                 for key, v in values.items()}
    summary = {"protocol": protocol, "selection": selection, "selected_teacher": selected,
               "runs": list(runs.values()), "seed_statistics": stats,
               "completed_at": datetime.datetime.now().isoformat()}
    audited = 0
    for r in runs.values():
        for split in ("validation", "test"):
            with np.load(OUT / r["name"] / "evaluation" / f"{split}_predictions.npz") as a:
                np.testing.assert_array_equal(a["ecg_ids"], frames[split].ecg_id)
                actual = metrics(a["targets"], a["probabilities"], a["thresholds"])
                for key in KEYS:
                    assert abs(actual[key] - r[split][key]) < 1e-12
                for label in SUPERCLASSES:
                    assert actual["per_class"][label] == r[split]["per_class"][label]
            audited += 1
    for relative, sha in protocol["protected_artifacts"].items():
        assert digest(ROOT / relative) == sha, f"Protected artifact changed: {relative}"
    normalization_checked = {}
    for r in runs.values():
        cfg = load(Path(r["checkpoint"]))["config"]
        if r["preprocessing"] not in normalization_checked:
            raw_config = replace(FilterConfig.from_mapping(cfg["signal_processing"]),
                                 normalization="none", training_mean=(), training_std=())
            raw_cache = ensure_preprocessed_cache(frames["train"], cfg["dataset"]["root"],
                          ROOT / cfg["outputs"]["cache_dir"], split_name="train_full",
                          sampling_rate_hz=100, config=raw_config)
            normalization_checked[r["preprocessing"]] = fit_training_normalization(raw_cache)
        expected_mean, expected_std = normalization_checked[r["preprocessing"]]
        np.testing.assert_allclose(cfg["signal_processing"]["training_mean"], expected_mean, rtol=0, atol=1e-12)
        np.testing.assert_allclose(cfg["signal_processing"]["training_std"], expected_std, rtol=0, atol=1e-12)
    assert digest(OUT / "selected_teacher.pt") == selected["checkpoint_sha256"]
    save(OUT / "audit.json", {"passed": True, "prediction_sets_recomputed": audited,
                              "per_class_metrics_recomputed": True, "patient_isolation_verified": True,
                              "training_only_normalization": True, "protected_artifacts_unchanged": True,
                              "selected_teacher_sha256": selected["checkpoint_sha256"]})
    save(OUT / "summary.json", summary)
    report(summary)
    status("Complete: teacher/preprocessing study, all-model metrics, paired seeds, audit, existing MD records updated")


if __name__ == "__main__":
    main()
