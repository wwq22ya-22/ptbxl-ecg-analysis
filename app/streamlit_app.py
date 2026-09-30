"""Interactive PTB-XL ECG viewer and research demonstration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ptbxl import (
    LEAD_NAMES, SUPERCLASSES, PTBXLDataError, encode_superclass_labels,
    find_ptbxl_root, load_ecg, load_ptbxl_metadata, load_scp_statements,
)
from src.config import load_config, resolve_project_path
from src.explainability import compute_input_saliency
from src.inference import load_predictor
from src.signal_processing import (
    FilterConfig, analyze_heart_rate, calculate_hrv, detect_r_peaks, preprocess_ecg,
)
from src.visualization.ecg_plot import (
    plot_12_leads, plot_raw_filtered_with_rpeaks, plot_rr_intervals,
)
from src.visualization.saliency_plot import plot_saliency_overlay

st.set_page_config(page_title="十二导联 ECG 分析", page_icon="🫀", layout="wide")


@st.cache_data(show_spinner=False)
def read_project_config() -> dict:
    return load_config(PROJECT_ROOT / "configs/default.yaml")


@st.cache_data(show_spinner="正在读取记录目录…")
def read_metadata(root: str, fs: int) -> pd.DataFrame:
    return load_ptbxl_metadata(root, sampling_rate_hz=fs)


@st.cache_data(show_spinner=False)
def read_statements(root: str, fs: int) -> pd.DataFrame:
    return load_scp_statements(root, sampling_rate_hz=fs)


@st.cache_resource(show_spinner="正在加载分类模型…")
def read_predictor(path: str, modified_ns: int):
    return load_predictor(path)


@st.cache_data(show_spinner="正在读取 ECG 波形…")
def read_record(root: str, filename: str, fs: int):
    return load_ecg(root, filename, sampling_rate_hz=fs)


def show_figure(figure) -> None:
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


def format_measurement(value: float | None, digits: int = 1) -> str:
    return "不可计算" if value is None else f"{value:.{digits}f}"


def main() -> None:
    config = read_project_config()
    fs = int(config["dataset"]["sampling_rate_hz"])
    st.title("十二导联 ECG 智能分析")
    st.caption("PTB-XL 教学与科研演示 · 自动分析可能误判，不用于临床诊断或治疗决策")

    try:
        root = find_ptbxl_root(
            resolve_project_path(config["dataset"]["root"], PROJECT_ROOT), sampling_rate_hz=fs
        )
        metadata = read_metadata(str(root), fs)
    except (PTBXLDataError, OSError, ValueError) as exc:
        st.error("未找到可读取的 PTB-XL 数据。请先按项目 README 的数据准备步骤放置数据。")
        st.code(str(exc))
        st.stop()

    st.sidebar.header("选择模型与 ECG")
    checkpoint_dir = PROJECT_ROOT / config["outputs"]["checkpoint_dir"]
    model_options = {
        "CNN 1D": ("cnn1d", checkpoint_dir / "baseline_best.pt"),
        "ResNet 1D": ("resnet1d", checkpoint_dir / "resnet1d_best.pt"),
    }
    available_models = [name for name, (_, path) in model_options.items() if path.is_file()]
    if available_models:
        selected_model_name = st.sidebar.selectbox("分类模型", available_models)
        selected_model_type, checkpoint_path = model_options[selected_model_name]
    else:
        selected_model_name = "CNN 1D"
        selected_model_type, checkpoint_path = model_options[selected_model_name]
    metric_suffix = "full" if selected_model_type == "cnn1d" else f"{selected_model_type}_full"
    predictor = None
    model_error = None
    if checkpoint_path.is_file():
        try:
            predictor = read_predictor(str(checkpoint_path), checkpoint_path.stat().st_mtime_ns)
            if predictor.sampling_rate_hz != fs:
                raise ValueError("模型采样率与当前数据配置不一致。")
        except (OSError, RuntimeError, ValueError) as exc:
            model_error = str(exc)
            predictor = None
    else:
        model_error = "尚无完整训练的模型。运行 scripts/train_baseline.py 后可查看分类结果。"

    ecg_id = int(st.sidebar.number_input(
        "ECG ID", min_value=int(metadata["ecg_id"].min()),
        max_value=int(metadata["ecg_id"].max()), value=int(metadata["ecg_id"].iloc[0]), step=1,
    ))
    selected = metadata.loc[metadata["ecg_id"].eq(ecg_id)]
    if selected.empty:
        st.warning(f"ECG ID {ecg_id} 不在 PTB-XL 记录目录中。请换一个编号。")
        st.stop()
    row = selected.iloc[0]
    filename_column = "filename_lr" if fs == 100 else "filename_hr"
    try:
        record = read_record(str(root), str(row[filename_column]), fs)
    except (PTBXLDataError, OSError, ValueError) as exc:
        st.error(f"ECG ID {ecg_id} 的波形无法读取：{exc}")
        st.stop()

    fold = int(row["strat_fold"])
    split_name = "训练" if fold <= 8 else "验证" if fold == 9 else "测试"
    st.sidebar.write(f"记录：{ecg_id} · {split_name}集（fold {fold}）")
    st.sidebar.write(f"采样率：{fs} Hz · 时长：{record.signal.shape[0] / fs:g} 秒")

    overview, viewer, analysis, classification, explanation, performance = st.tabs(
        ["概览", "十二导联", "信号分析", "AI 分类结果", "模型关注区域", "模型性能"]
    )
    with overview:
        st.subheader("当前记录")
        columns = st.columns(4)
        columns[0].metric("ECG ID", str(ecg_id))
        columns[1].metric("导联", "12")
        columns[2].metric("采样率", f"{fs} Hz")
        columns[3].metric("模型", selected_model_name if predictor else "未加载")
        st.write(f"数据集：PTB-XL 1.0.3，共 {len(metadata):,} 条记录。当前记录属于{split_name}集。")
        st.write("本页面可查看十二导联波形、R 峰与心率、短记录 HRV，以及五类多标签模型预测。")
        if model_error:
            st.info(model_error)
        try:
            statements = read_statements(str(root), fs)
            reference = encode_superclass_labels(row["scp_codes"], statements)
            names = [name for index, name in enumerate(SUPERCLASSES) if reference[index]]
            st.caption("数据集参考标签：" + ("、".join(names) if names else "五类中无对应诊断标签"))
        except (PTBXLDataError, OSError, ValueError) as exc:
            st.caption(f"参考标签暂不可用：{exc}")

    with viewer:
        st.subheader("十二导联原始波形")
        st.caption("横轴为秒；纵轴使用 WFDB 记录中的物理单位。")
        show_figure(plot_12_leads(record.signal, fs, units=record.units, title=f"ECG {ecg_id}"))

    with analysis:
        st.subheader("Lead II 信号分析")
        try:
            settings = config["signal_processing"]
            filtered = preprocess_ecg(
                record.signal, fs, config=FilterConfig.from_mapping(settings), normalization="none"
            )
            lead_ii = LEAD_NAMES.index("II")
            peaks = detect_r_peaks(
                filtered[:, lead_ii], fs,
                minimum_rr_s=float(settings["rpeak_minimum_rr_s"]),
                integration_window_s=float(settings["rpeak_integration_window_s"]),
                t_wave_window_s=float(settings["rpeak_t_wave_window_s"]),
                t_wave_max_ratio=float(settings["rpeak_t_wave_max_ratio"]),
            )
            heart_rate = analyze_heart_rate(
                peaks, fs,
                min_valid_rr_s=float(settings["rr_min_valid_s"]),
                max_valid_rr_s=float(settings["rr_max_valid_s"]),
            )
            hrv = calculate_hrv(
                heart_rate.valid_rr_intervals_s, record_duration_s=record.signal.shape[0] / fs
            )
            show_figure(plot_raw_filtered_with_rpeaks(
                record.signal, filtered, fs, lead="II", r_peak_indices=peaks,
                unit=record.units[lead_ii],
            ))
            hr_columns = st.columns(3)
            hr_columns[0].metric("平均心率", format_measurement(heart_rate.mean_hr_bpm) + " bpm")
            hr_columns[1].metric("最小心率", format_measurement(heart_rate.min_hr_bpm) + " bpm")
            hr_columns[2].metric("最大心率", format_measurement(heart_rate.max_hr_bpm) + " bpm")
            st.caption(f"检测到 {len(peaks)} 个 R 峰；有效 RR 间期 {len(heart_rate.valid_rr_intervals_s)} 个；排除 {heart_rate.excluded_rr_count} 个异常 RR 间期。")
            if heart_rate.message:
                st.info(heart_rate.message)
            show_figure(plot_rr_intervals(heart_rate.valid_rr_intervals_s))
            hrv_columns = st.columns(3)
            hrv_columns[0].metric("Mean RR", format_measurement(hrv.mean_rr_ms) + " ms")
            hrv_columns[1].metric("SDNN", format_measurement(hrv.sdnn_ms) + " ms")
            hrv_columns[2].metric("RMSSD", format_measurement(hrv.rmssd_ms) + " ms")
            if hrv.message:
                st.info(hrv.message)
            if hrv.warning:
                st.warning(hrv.warning)
        except (ValueError, RuntimeError) as exc:
            st.error(f"这条记录的信号分析未能完成：{exc}")

    with classification:
        st.subheader(f"五类模型预测 · {selected_model_name}")
        if predictor is None:
            st.info(model_error or "分类模型暂不可用。")
        else:
            try:
                result = predictor.predict_record(record)
            except (ValueError, RuntimeError) as exc:
                st.error(f"这条记录无法分类：{exc}")
            else:
                chart = pd.DataFrame({"类别": list(SUPERCLASSES), "概率": list(result.probabilities.values())})
                st.bar_chart(chart.set_index("类别"))
                table = pd.DataFrame({
                    "类别": list(SUPERCLASSES),
                    "概率": [f"{result.probabilities[name]:.3f}" for name in SUPERCLASSES],
                    "阈值": [f"{result.thresholds[name]:.2f}" for name in SUPERCLASSES],
                    "超过阈值": ["是" if name in result.positive_labels else "否" for name in SUPERCLASSES],
                })
                st.dataframe(table, hide_index=True, width="stretch")
                st.write("超过阈值的类别：" + ("、".join(result.positive_labels) if result.positive_labels else "无"))
                st.caption("模型预测与数据集参考标签是不同来源；概率未经外部人群校准，不能据此做临床判断。")

    with explanation:
        st.subheader("模型关注区域")
        if predictor is None:
            st.info(model_error or "分类模型暂不可用。")
        else:
            selected_class = st.selectbox("查看类别", list(SUPERCLASSES))
            try:
                saliency = compute_input_saliency(predictor, record, class_name=selected_class)
                show_figure(plot_saliency_overlay(record, saliency, lead="II"))
                st.caption("上方显示 Lead II 原始波形；下方曲线汇总十二导联对该类别输出的相对响应。颜色和高度仅用于这条记录内的可视化比较。")
                st.warning("显著性图只反映模型内部响应，不等于临床因果解释或医生判断依据。")
            except (ValueError, RuntimeError) as exc:
                st.error(f"模型关注区域暂不可用：{exc}")

    with performance:
        st.subheader("独立测试集表现")
        metrics_path = PROJECT_ROOT / config["outputs"]["metrics_dir"] / f"test_metrics_{metric_suffix}.json"
        if not metrics_path.is_file():
            st.info("尚无正式测试结果。先运行 scripts/evaluate_model.py。")
        else:
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                summary = st.columns(3)
                summary[0].metric("测试记录", f"{metrics['records']:,}")
                summary[1].metric("Macro AUROC", f"{metrics['macro_auroc']:.3f}")
                summary[2].metric("Macro F1", f"{metrics['macro_f1']:.3f}")
                rows = []
                for name in SUPERCLASSES:
                    value = metrics["per_class"][name]
                    rows.append({
                        "类别": name, "AUROC": value["auroc"], "Precision": value["precision"],
                        "Recall": value["recall"], "F1": value["f1"],
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                figure_root = PROJECT_ROOT / config["outputs"]["figure_dir"]
                for filename, caption in (
                    (f"test_roc_{metric_suffix}.png", "各类 ROC 曲线"),
                    (f"test_confusion_{metric_suffix}.png", "每类二分类混淆矩阵"),
                    ("training_curve_full.png" if selected_model_type == "cnn1d" else f"training_curve_{selected_model_type}_full.png", "训练与验证曲线"),
                    ("class_distribution_full.png" if selected_model_type == "cnn1d" else f"class_distribution_{selected_model_type}_full.png", "训练集类别分布"),
                ):
                    image_path = figure_root / filename
                    if image_path.is_file():
                        st.image(str(image_path), caption=caption, width="stretch")
                    else:
                        st.info(f"{caption}尚未生成；请检查训练或评估输出。")
                st.caption("HYP 类表现较弱；页面展示的指标只适用于本次 PTB-XL 测试，不代表临床使用效果。")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.error(f"测试结果文件无法读取：{exc}")


if __name__ == "__main__":
    main()
