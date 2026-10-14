"""Write conclusions from the completed, audited experiment into existing MDs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/teacher_preprocessing"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def section(path, body, marker):
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    previous = path.read_text(encoding="utf-8")
    new = start + "\n" + body + "\n" + end
    if start in previous:
        left, rest = previous.split(start, 1)
        _, right = rest.split(end, 1)
        updated = left + new + right
    else:
        updated = previous.rstrip() + "\n\n" + new + "\n"
    path.write_text(updated, encoding="utf-8")


def plot_pairs(rows, family):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    labels = ["NORM", "MI", "STTC", "CD", "HYP"]
    groups = [("inception1d", family[1], "紧凑 Inception"),
              (family[0], family[1], "InceptionTime 教师" if family[0] == "inception_time1d" else "SE-ResNet18 教师")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for index, (model, preprocessing, name) in enumerate(groups):
        members = [r for r in rows if (r["model"], r["preprocessing"]) == (model, preprocessing)]
        assert len(members) == 3
        overall = np.array([[r["test"][k] for k in ("macro_precision", "macro_recall", "macro_f1")]
                            for r in members]) * 100
        recall = np.array([[r["test"]["per_class"][label]["recall"] for label in labels]
                           for r in members]) * 100
        offset = -.18 if index == 0 else .18
        color = ("#4C78A8", "#F58518")[index]
        for ax, values, ticks in zip(axes, (overall, recall),
                                     (["Precision", "Recall", "F1"], labels)):
            x = np.arange(len(ticks))
            bars = ax.bar(x + offset, values.mean(0), width=.34, label=name, color=color,
                          yerr=values.std(0, ddof=1), capsize=3, zorder=3)
            ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=6)
            ax.set_xticks(x, ticks)
            ax.set_ylim(0, 103)
            ax.set_ylabel("指标（%）")
            ax.grid(axis="y", alpha=.2, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("宏平均指标")
    axes[1].set_title("各类召回率")
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("同预处理 FP32 教师对照：三个种子均值，误差线为样本标准差", fontsize=12)
    fig.savefig(OUT / "paired_teacher_comparison.png", dpi=180)
    plt.close(fig)


def main():
    summary = read(OUT / "summary.json")
    audit = read(OUT / "independent_audit.json")
    bootstrap = read(OUT / "paired_patient_bootstrap.json")
    assert audit["passed"]
    chosen = summary["selected_teacher"]
    baseline_family = "inception1d/" + chosen["preprocessing"]
    new_family = chosen["model"] + "/" + chosen["preprocessing"]
    base = summary["seed_statistics"][baseline_family]
    new = summary["seed_statistics"][new_family]
    current = read(ROOT / "outputs/noise_robustness_b/selected_export/A_deployed_seed44_INT8_robustness.json")["test"]["clean"]
    rows = summary["runs"]
    plot_pairs(rows, (chosen["model"], chosen["preprocessing"]))
    teacher_mean = new["macro_f1"]["mean"]
    delta = teacher_mean - base["macro_f1"]["mean"]
    precision_delta = new["macro_precision"]["mean"] - base["macro_precision"]["mean"]
    recall_delta = new["macro_recall"]["mean"] - base["macro_recall"]["mean"]
    preprocessing_scores = "、".join(f"{r['preprocessing']} {r['validation']['macro_f1']:.6f}"
        for r in rows if r["model"] == "inception1d" and r["seed"] == 42)
    findings = ["### 本轮结论与可用教师", "",
        f"完成 {len(rows)} 次完整训练。预处理首轮（seed42）验证宏F1：{preprocessing_scores}；按预先约定的验证规则选择{summary['selection']['preprocessing']}。两个替代预处理仅作单种子筛选，小差异不构成它们普遍无效的证明。",
        f"同预处理、同训练预算的三种子平均测试F1：紧凑Inception {base['macro_f1']['mean']:.6f} ± {base['macro_f1']['std']:.6f}；新教师 {teacher_mean:.6f} ± {new['macro_f1']['std']:.6f}，变化 {delta*100:+.3f} 个百分点。",
        f"三种子平均Precision变化 {precision_delta*100:+.3f} 个百分点，Recall变化 {recall_delta*100:+.3f} 个百分点；只有二者都改善才能称本轮平均精确率和召回率一起提高。",
        "", "![教师三种子对照](../outputs/teacher_preprocessing/paired_teacher_comparison.png)", "",
        "| 种子 | 新教师−同预处理紧凑Inception：Precision | Recall | F1 |", "|---|---:|---:|---:|"]
    for seed in (42, 43, 44):
        control = next(r for r in rows if (r["model"], r["preprocessing"], r["seed"]) ==
                       ("inception1d", chosen["preprocessing"], seed))
        teacher = next(r for r in rows if (r["model"], r["preprocessing"], r["seed"]) ==
                       (chosen["model"], chosen["preprocessing"], seed))
        values = [(teacher["test"][key] - control["test"][key]) * 100
                  for key in ("macro_precision", "macro_recall", "macro_f1")]
        findings.append(f"| {seed} | " + " | ".join(f"{v:+.3f} 个百分点" for v in values) + " |")
    findings += ["", "| 对照差值：新教师−紧凑Inception三种子均值 | 观测变化 | 患者配对重采样95%区间 |", "|---|---:|---:|"]
    for key, label in (("macro_precision", "Precision"), ("macro_recall", "Recall"),
                       ("macro_f1", "F1"), ("hyp_recall", "HYP Recall")):
        b = bootstrap["differences"][key]
        lo, hi = b["percentile_95_interval"]
        findings.append(f"| {label} | {b['mean_delta']*100:+.3f} 个百分点 | [{lo*100:+.3f}, {hi*100:+.3f}] 个百分点 |")
    findings += ["", "上述区间为500次患者簇配对重采样，使用固定的三对模型和验证阈值，是探索性的不确定性检查；不包含重新训练的全部变异，也不是新的外部验证。",
        f"验证集冻结的最终教师：{chosen['name']}，参数{chosen['parameters']}，最佳轮次{chosen['epoch']}；文件为 `outputs/teacher_preprocessing/selected_teacher.pt`。没有根据测试得分改选种子。",
        "", "| 固定测试集上的单模型部署参考 | Precision | Recall | F1 |", "|---|---:|---:|---:|",
        "| 原部署 seed44 INT8 | " + " | ".join(f"{current[k]:.6f}" for k in ("macro_precision", "macro_recall", "macro_f1")) + " |",
        f"| 验证集选定的新教师 FP32（seed{chosen['seed']}） | " + " | ".join(f"{chosen['test'][k]:.6f}" for k in ("macro_precision", "macro_recall", "macro_f1")) + " |",
        "", "单模型参考表的两个模型架构和精度格式不同，不能把教师成绩直接当作新INT8学生成绩。当前部署仍为原seed44；本轮没有重做学生蒸馏、量化或固件替换。",
        f"独立审计通过：{audit['prediction_sets_independently_verified']}组预测、所有整体/逐类指标和混淆矩阵复算一致；{audit['normalization_independently_recomputed']}种预处理的训练集标准化独立复算；{audit['raw_preprocessing_records_rechecked']}条原始记录重新滤波一致。患者隔离、验证选择重放、原模型/固件SHA256检查通过。",
        "复核：`.venv/Scripts/python.exe scripts/audit_teacher_preprocessing.py`；更新结论：`.venv/Scripts/python.exe scripts/summarize_teacher_study.py`。"]
    section(ROOT / "md/模型训练与优化.md", "\n".join(findings), "teacher-study-findings")
    entry = (f"2026-10-07 完成更强教师与预处理消融，共{len(rows)}次训练，包含入选教师与紧凑Inception的三个种子配对复核。"
             f"预处理按验证集规则保留原方案；教师候选为 `{chosen['name']}`，权重另存 `outputs/teacher_preprocessing/selected_teacher.pt`。"
             "当前部署仍为原seed44 INT8。整体/逐类结果已追加到现有训练文档和指标总表，配置、预测及独立审计保存在 `outputs/teacher_preprocessing/`。")
    section(ROOT / "README.md", entry, "teacher-study-entry")
    section(ROOT / "md/README.md", entry, "teacher-study-entry")
    print(json.dumps({"teacher": chosen["name"], "parameters": chosen["parameters"],
                      "teacher_test": chosen["test"], "current_int8": current,
                      "paired_mean_f1_delta": delta}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
