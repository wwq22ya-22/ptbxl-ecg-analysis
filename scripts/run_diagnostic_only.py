"""Train one fresh ResNet after removing records with no five-class diagnosis."""
from __future__ import annotations
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest
from src.data.ptbxl import SUPERCLASSES
from src.models import build_model
from src.signal_processing.filters import FilterConfig
from src.training.data import (
    CachedPTBXLDataset, ensure_cache_subset, ensure_preprocessed_cache,
    fit_training_normalization, load_labeled_splits,
)
from src.training.metrics import calculate_multilabel_metrics, optimize_thresholds
from src.training.train import train_baseline
from src.training.evaluate import evaluate_checkpoint

OUT = ROOT / "outputs/diagnostic_only"
OLD = ROOT / "outputs/optimization/01_global_norm/checkpoint/resnet1d_best.pt"
NEW = OUT / "resnet_seed42/checkpoint/resnet1d_best.pt"
REPORT = ROOT / "outputs/reports/剔除无五类标签样本_重训练报告.md"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def status(stage):
    save(OUT / "status.json", {"stage": stage, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(stage, flush=True)


def split_summary(frame):
    targets = frame[list(SUPERCLASSES)].to_numpy()
    return {"records": len(frame), "patients": int(frame.patient_id.nunique()),
            "all_zero_records": int((targets.sum(axis=1) == 0).sum()),
            "positives": {name: int(targets[:, i].sum()) for i, name in enumerate(SUPERCLASSES)}}


def predict(path, selected, full, split_name):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source = selected if checkpoint["config"]["dataset"].get("require_diagnostic_label", False) else full
    cfg = checkpoint["config"]
    cache = ensure_preprocessed_cache(source, Path(cfg["dataset"]["root"]), ROOT / cfg["outputs"]["cache_dir"],
                                     split_name=f"{split_name}_full", sampling_rate_hz=100,
                                     config=FilterConfig.from_mapping(cfg["signal_processing"]))
    positions = pd.Index(source.ecg_id).get_indexer(selected.ecg_id)
    if (positions < 0).any(): raise ValueError("Comparison records are missing from model inputs.")
    np.testing.assert_array_equal(source.iloc[positions][list(SUPERCLASSES)], selected[list(SUPERCLASSES)])
    loader = DataLoader(Subset(CachedPTBXLDataset(cache, source), positions.tolist()), batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint["model_type"], dropout=cfg["model"]["dropout"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    with torch.inference_mode():
        scores = np.concatenate([torch.sigmoid(model(x.to(device))).cpu().numpy() for x, _ in loader])
    del model
    return scores, checkpoint


def write_report(summary):
    cohort = summary["cohort"]
    lines = ["# 剔除无五类诊断标签样本后的重训练报告", "", "日期：2026-10-01", "",
             "## 本轮规则", "",
             "删除 NORM、MI、STTC、CD、HYP 五列全部为 0 的记录；只要包含任一五类诊断标签，就予以保留。"
             "例如 PACE + MI 会保留，只标 PACE 的记录会剔除。原始 PhysioNet 文件保留，训练使用独立的筛选后样本清单。", "",
             "## 数据划分", "", "| 划分 | 原记录数 | 剔除 | 新记录数 | 新患者数 | 五类全零 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, part in cohort.items():
        a, b = part["before"], part["after"]
        lines.append(f"| {name} | {a['records']} | {a['records']-b['records']} | {b['records']} | {b['patients']} | {b['all_zero_records']} |")
    lines += ["", "仍使用官方训练 folds 1–8、验证 fold 9、测试 fold 10，患者集合没有交叉。"
              "筛选只依据原始五类标注，未按预测结果删除难样本。各类阳性数保持不变。", "",
              "## 新训练设置", "",
              "从随机初始化重新训练标准化 ResNet1D，参数 450885；没有载入旧权重继续微调。"
              "种子 42，batch size 32，AdamW，学习率 0.001，weight decay 0.0001，BCE 正类权重，"
              "最多 20 轮，early stopping patience 5，按验证 AUROC 选最佳权重。"
              "网络结构和训练超参数沿用旧标准化 ResNet；重新计算剔除后训练集的标准化统计与类别权重。", "",
              f"最佳 epoch：{summary['new_model']['epoch']}；实际训练时间：{summary['training_seconds']:.1f} 秒。", "",
              "逐类阈值只在筛选后的验证折 9 上用原 0.10–0.90、步长 0.05 的粗网格选择。"
              "新模型阈值冻结后才评估测试集；没有根据测试结果选择种子、轮数或阈值。", "",
              "## 相同测试样本的整体对比", "",
              "以下所有版本都在同一批 **2158 条、至少有一个五类诊断标签** 的 fold-10 记录上评估。"
              "不能直接与历史 2198 条全样本测试结果作训练收益比较。", "",
              "| 模型 / 阈值策略 | AUROC | Precision | Recall | F1 |",
              "|---|---:|---:|---:|---:|"]
    for result in summary["comparisons"]:
        m = result["test"]
        lines.append("| " + result["name"] + " | " + " | ".join(f"{m[k]:.6f}" for k in
                     ["macro_auroc", "macro_precision", "macro_recall", "macro_f1"]) + " |")
    control = next(r for r in summary["comparisons"] if r["id"] == "old_resnet_recalibrated")
    new = next(r for r in summary["comparisons"] if r["id"] == "new_resnet")
    lines += ["", "主对照是旧 ResNet 在新验证集上重新校准阈值后的版本与新训练 ResNet。"
              "二者使用同一架构、同一阈值搜索方法和相同测试样本；旧模型仍保留原训练权重与原训练标准化统计。", "",
              "| 指标 | 新训练 − 旧 ResNet 重新校准 |", "|---|---:|"]
    for key in ["macro_auroc", "macro_precision", "macro_recall", "macro_f1"]:
        lines.append(f"| {key} | {new['test'][key]-control['test'][key]:+.6f} |")
    change = new["test"]["macro_f1"] - control["test"]["macro_f1"]
    lines += ["", f"本次单种子重训练的宏 F1 {'提高' if change > 0 else '下降' if change < 0 else '持平'}"
              f" {abs(change)*100:.3f} 个百分点。筛选改变了数据集、标准化统计和正类权重；"
              "单次训练不证明收益稳定，也不能将所有差异归因于某一种因素。", "",
              "## 每个对照模型的逐类测试指标", "",
              "| 模型 / 阈值策略 | 类别 | 阳性数 | AUROC | Precision | Recall | F1 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for result in summary["comparisons"]:
        for name in SUPERCLASSES:
            m = result["test"]["per_class"][name]
            lines.append(f"| {result['name']} | {name} | {m['support']} | " +
                         " | ".join(f"{m[k]:.6f}" for k in ["auroc", "precision", "recall", "f1"]) + " |")
    lines += ["", "## 验证结果与阈值", "",
              "| 模型 | 验证记录数 | AUROC | Precision | Recall | F1 | 阈值 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for result in summary["comparisons"]:
        if "validation" not in result: continue
        m = result["validation"]
        lines.append(f"| {result['name']} | {m['records']} | " + " | ".join(f"{m[k]:.6f}" for k in
                     ["macro_auroc", "macro_precision", "macro_recall", "macro_f1"]) +
                     " | " + ", ".join(f"{v:.3f}" for v in result["thresholds"]) + " |")
    lines += ["", "## 审计与文件", "",
              "训练、验证、测试保留的记录号及原始 SCP 标注分别在 `outputs/diagnostic_only/cohort_*.csv`；"
              "剔除清单在 `removed_records.csv`，共 411 条。没有因为携带节律或形态标注而一律删除记录。", "",
              "- 新检查点：[resnet1d_best.pt](../diagnostic_only/resnet_seed42/checkpoint/resnet1d_best.pt)。",
              "- 实验配置：[config.json](../diagnostic_only/config.json)。",
              "- 整体、逐类指标与审计：[summary.json](../diagnostic_only/summary.json)。",
              "- 逐条预测：`outputs/diagnostic_only/comparison_predictions/`。",
              "- 训练日志：[diagnostic_only_training.log](../diagnostic_only_training.log)。", "",
              "本轮新模型尚未量化、生成 Cube.AI 固件或验证串口。它的标准化参数和阈值与旧模型可能不同，"
              "部署需要为新检查点重新生成对应文件。筛选后模型仍只判断五个诊断大类，不能自动识别 PACE、AFIB 等独立节律标签。", "",
              "已有模型曾在旧测试样本上多次比较，本轮与历史测试集有大量重叠，结果不能当成全新的盲测数据。"
              "本轮完成的是一次事先固定配置的新训练及同样本对比，不进行基于测试结果的追加调参。", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "summary.json").exists():
        write_report(json.loads((OUT / "summary.json").read_text(encoding="utf-8")))
        print("Completed run already exists; report refreshed.", flush=True)
        return
    reference = torch.load(OLD, map_location="cpu", weights_only=True)
    original_hash = digest(OLD)
    cfg = copy.deepcopy(reference["config"])
    cfg["dataset"]["require_diagnostic_label"] = True
    for key in ("training_mean", "training_std"): cfg["signal_processing"].pop(key, None)
    for key, folder in [("checkpoint_dir", "checkpoint"), ("metrics_dir", "metrics"),
                        ("figure_dir", "figure"), ("predictions_dir", "predictions")]:
        cfg["outputs"][key] = f"outputs/diagnostic_only/resnet_seed42/{folder}"
    save(OUT / "config.json", cfg)
    dataset = Path(cfg["dataset"]["root"])
    full = load_labeled_splits(dataset, 100)
    selected = load_labeled_splits(dataset, 100, require_diagnostic_label=True)
    cohort = {}
    removed = []
    for name in full:
        a, b = full[name], selected[name]
        assert (b[list(SUPERCLASSES)].sum(axis=1) > 0).all()
        cohort[name] = {"before": split_summary(a), "after": split_summary(b)}
        assert cohort[name]["before"]["positives"] == cohort[name]["after"]["positives"]
        b.to_csv(OUT / f"cohort_{name}.csv", index=False)
        removed.append(a.loc[~a.ecg_id.isin(b.ecg_id), ["ecg_id", "patient_id", "strat_fold", "scp_codes"]].assign(split=name))
    pd.concat(removed).to_csv(OUT / "removed_records.csv", index=False)
    save(OUT / "cohort_summary.json", cohort)
    print("Cohorts:", cohort, flush=True)
    status("Preparing checked cache subsets and new training-only normalization")
    raw_cfg = FilterConfig.from_mapping({**cfg["signal_processing"], "normalization": "none"})
    cache_dir = ROOT / cfg["outputs"]["cache_dir"]
    for name in full:
        source = ensure_preprocessed_cache(full[name], dataset, cache_dir, split_name=f"{name}_full",
                                           sampling_rate_hz=100, config=raw_cfg)
        ensure_cache_subset(source, full[name], selected[name], cache_dir, split_name=f"{name}_full",
                            sampling_rate_hz=100, config=raw_cfg)
    history = OUT / "resnet_seed42/metrics/training_history_resnet1d_full.json"
    started = time.monotonic()
    if NEW.exists() and not history.exists():
        raise RuntimeError("An interrupted run exists; preserve and inspect it instead of silently using its partial checkpoint.")
    if not NEW.exists():
        status("Training fresh diagnostic-only ResNet, fixed seed 42")
        checkpoint_path = train_baseline(cfg, project_root=ROOT, dataset_root=dataset, model_type="resnet1d")
    else: checkpoint_path = NEW
    seconds = time.monotonic() - started
    new_cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert new_cp["training_records"] == 17084 and new_cp["validation_records"] == 2146
    raw_train = ensure_preprocessed_cache(selected["train"], dataset, cache_dir, split_name="train_full",
                                          sampling_rate_hz=100, config=raw_cfg)
    mean, std = fit_training_normalization(raw_train)
    np.testing.assert_allclose(mean, new_cp["config"]["signal_processing"]["training_mean"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(std, new_cp["config"]["signal_processing"]["training_std"], atol=1e-12, rtol=0)
    status("Freezing thresholds from filtered validation fold 9 before test comparison")
    old_val, old_cp = predict(OLD, selected["validation"], full["validation"], "validation")
    truth_val = selected["validation"][list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    old_recalibrated = optimize_thresholds(truth_val, old_val)
    save(OUT / "frozen_threshold_policy.json", {"new": new_cp["thresholds"],
         "old_resnet_fixed": old_cp["thresholds"], "old_resnet_recalibrated": old_recalibrated,
         "source": "filtered validation fold 9 only", "test_used_for_threshold_selection": False})
    status("Evaluating frozen models on the same 2158 filtered fold-10 records")
    new_test = evaluate_checkpoint(checkpoint_path, project_root=ROOT, dataset_root=dataset)
    old_test, _ = predict(OLD, selected["test"], full["test"], "test")
    truth_test = selected["test"][list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    predictions = OUT / "comparison_predictions"
    predictions.mkdir(exist_ok=True)
    np.savez_compressed(predictions / "old_resnet.npz", targets=truth_test, probabilities=old_test,
                        ecg_ids=selected["test"].ecg_id.to_numpy())
    results = []
    for key, name, thresholds in [("old_resnet_fixed", "旧标准化 ResNet / 原阈值", old_cp["thresholds"]),
                                 ("old_resnet_recalibrated", "旧标准化 ResNet / 新验证集重校准", old_recalibrated)]:
        results.append({"id": key, "name": name, "checkpoint": str(OLD), "thresholds": thresholds,
                        "validation": calculate_multilabel_metrics(truth_val, old_val, thresholds),
                        "test": calculate_multilabel_metrics(truth_test, old_test, thresholds)})
    results.append({"id": "new_resnet", "name": "剔除后新训练 ResNet", "checkpoint": str(checkpoint_path),
                    "thresholds": new_cp["thresholds"], "validation": new_cp["validation_metrics_tuned"], "test": new_test})
    inception = ROOT / "outputs/optimization/optimized_best.pt"
    other_test, other_cp = predict(inception, selected["test"], full["test"], "test")
    results.append({"id": "old_inception", "name": "旧优化 Inception / 原冻结阈值", "checkpoint": str(inception),
                    "thresholds": other_cp["thresholds"],
                    "test": calculate_multilabel_metrics(truth_test, other_test, other_cp["thresholds"])})
    np.savez_compressed(predictions / "old_inception.npz", targets=truth_test, probabilities=other_test,
                        ecg_ids=selected["test"].ecg_id.to_numpy())
    ensemble_path = ROOT / "outputs/further_optimization/threshold_refinement/ensemble_manifest.json"
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    ensemble_scores = np.zeros_like(old_test)
    for component in ensemble["components"]:
        source = Path(component["checkpoint"])
        if digest(source) != component["sha256"]: raise ValueError("Frozen ensemble component changed.")
        scores, _ = predict(source, selected["test"], full["test"], "test")
        ensemble_scores += component["weight"] * scores
    results.append({"id": "old_ensemble", "name": "旧等权集成 / 原冻结阈值", "checkpoint": str(ensemble_path),
                    "thresholds": ensemble["thresholds"],
                    "test": calculate_multilabel_metrics(truth_test, ensemble_scores, ensemble["thresholds"])})
    np.savez_compressed(predictions / "old_ensemble.npz", targets=truth_test, probabilities=ensemble_scores,
                        ecg_ids=selected["test"].ecg_id.to_numpy())
    if digest(OLD) != original_hash: raise ValueError("The original checkpoint was modified.")
    summary = {"cohort": cohort, "training_seconds": seconds,
               "new_model": {"checkpoint": str(checkpoint_path), "sha256": digest(checkpoint_path),
                             "epoch": new_cp["epoch"], "training_records": new_cp["training_records"],
                             "validation_records": new_cp["validation_records"]},
               "comparisons": results, "old_checkpoint_unchanged": True,
               "normalization_training_only_verified": True, "test_used_for_selection": False}
    save(OUT / "summary.json", summary)
    write_report(summary)
    status("Completed: one new model trained, fair comparison and Markdown report saved")
    for result in results:
        print(result["name"], {k: v for k, v in result["test"].items() if k.startswith("macro")}, flush=True)


if __name__ == "__main__": main()
