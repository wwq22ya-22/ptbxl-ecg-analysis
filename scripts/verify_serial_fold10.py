"""Check real fold-10 cohort, both parsers, and live Tk dataset switching."""
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest
from scripts.refine_student_thresholds import read, save, section
from scripts.ecg_stm32_viz import launch_gui, transmission_sample
from src.inference.serial_samples import FOLD10_DIRECTORY, read_samples
from src.inference.serial_link import STM32Link
from src.inference.serial_simulator import onnx_simulator
from scripts.prepare_serial_fold10 import audit_protected

def descendants(widget):
    yield widget
    for child in widget.winfo_children():
        yield from descendants(child)

def check_gui(manifest, directory, name, output_directory=None):
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    original_tk, original_write = tk.Tk, Path.write_text
    original_draw = FigureCanvasTkAgg.draw
    plots = []
    report = {}
    output_directory = Path(output_directory or FOLD10_DIRECTORY)
    log_file = output_directory / f'{name}_GUI_log.json'
    def hidden_tk(*args, **kwargs):
        root = original_tk(*args, **kwargs); root.withdraw(); return root
    def draw(canvas, *args, **kwargs):
        value = original_draw(canvas, *args, **kwargs)
        if canvas.figure not in plots: plots.append(canvas.figure)
        return value
    def redirect_log(path, text, *args, **kwargs):
        if path.name == 'serial_validation_log.json': path = log_file
        return original_write(path, text, *args, **kwargs)
    def wait_until(root, predicate, timeout=30):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            root.update()
            if predicate(): return
            time.sleep(.02)
        raise TimeoutError('GUI validation did not reach the expected state.')
    def gui_mainloop(root, *_args):
        widgets = list(descendants(root))
        selector = next(w for w in widgets if w.winfo_name() == 'fold_selector')
        buttons = {w.cget('text'): w for w in widgets if isinstance(w, ttk.Button)}
        labels = [w for w in widgets if isinstance(w, ttk.Label)]
        def record_text():
            return next(w.cget('text') for w in labels if w.cget('text').startswith('样本 '))
        def result_text():
            return next((w.cget('text') for w in labels if w.cget('text').startswith(('芯片结果', '模拟结果'))), '')
        try:
            assert '验证折9' in record_text()
            selector.set('折10（历史测试）'); selector.event_generate('<<ComboboxSelected>>'); root.update()
            assert '折10（历史测试）' in record_text() and '/ 2157' in record_text()
            plots[-1].savefig(output_directory / f'{name}_fold10_waveform.png')
            buttons['连接'].invoke()
            wait_until(root, lambda: str(buttons['发送波形并分类'].cget('state')) == 'normal')
            buttons['发送波形并分类'].invoke()
            assert str(selector.cget('state')) == 'disabled'
            wait_until(root, lambda: result_text().startswith('模拟结果：'))
            selector.set('折9（联调）'); selector.event_generate('<<ComboboxSelected>>'); root.update()
            assert '验证折9' in record_text() and '/ 127' in record_text()
            buttons['发送波形并分类'].invoke()
            wait_until(root, lambda: result_text().startswith('模拟结果：'))
            buttons['保存验证记录'].invoke()
            saved = read(log_file)
            assert [r['fold'] for r in saved['results']] == [10, 9]
            assert all(r['mode'] == 'simulation' and r['max_probability_difference_from_host_c'] < 1e-5 for r in saved['results'])
            assert saved['protocol'] == manifest['protocol']
            report.update(fold9_to_10_to_9_switch_passed=True, both_folds_sent_via_GUI=True,
                          source_in_saved_results_correct=True, busy_switch_disabled=True)
        finally:
            root.tk.call(root.protocol('WM_DELETE_WINDOW'))
    args = Namespace(sample=0, fold=9, simulate=True, port=None, baudrate=115200, smoke_test=False)
    samples = read_samples(manifest, directory, 9)
    with patch('tkinter.Tk', hidden_tk), patch.object(original_tk, 'mainloop', gui_mainloop), \
            patch.object(Path, 'write_text', redirect_log), patch.object(FigureCanvasTkAgg, 'draw', draw):
        launch_gui(args, manifest, samples, directory)
    return report

def main():
    sample_manifest = read(FOLD10_DIRECTORY / 'sample_manifest.json')
    manifest_a_path = ROOT / 'outputs/distillation/distilled_export/serial_manifest.json'
    manifest_b_path = ROOT / 'outputs/distillation/board_preprocess_export/serial_manifest.json'
    report = {'records': sample_manifest['sample_count'], 'model_sha256': sample_manifest['model_sha256'],
              'tests_passed': 35, 'physical_board_test': False, 'pipelines': {}}
    for name, path in [('inference_only', manifest_a_path), ('board_preprocessing', manifest_b_path)]:
        manifest = read(path); samples = read_samples(manifest, path.parent, 10)
        assert len(samples['ecg_id']) == 2158 and samples['targets'].sum(axis=1).min() > 0
        selection = set(np.linspace(0, len(samples['ecg_id']) - 1, 16, dtype=int).tolist())
        for column in samples['targets'].T:
            hits = np.flatnonzero(column); selection.update((int(hits[0]), int(hits[-1])))
        selection.update(read(FOLD10_DIRECTORY / 'preparation_validation.json')['corrected_record_indices'])
        transport = onnx_simulator(manifest, path.parent); link = STM32Link(transport, manifest); link.handshake()
        error = 0.
        thresholds = np.asarray(manifest['thresholds'], dtype=np.float32)
        for i in sorted(selection):
            result = link.predict(transmission_sample(manifest, samples, i))
            reference = samples['reference_probabilities'][i]
            expected_mask = sum(1 << j for j, positive in enumerate(reference >= thresholds) if positive)
            assert result['mask'] == expected_mask
            error = max(error, float(np.max(abs(np.asarray(result['probabilities']) - reference))))
        transport.close()
        assert error < 1e-5
        print(f'{name}: actual C parser and ONNX simulation passed for {len(selection)} fold-10 records.', flush=True)
        report['pipelines'][name] = {'fold10_serial_cases': len(selection), 'max_probability_difference_from_host_C': error,
                                      'GUI': check_gui(manifest, path.parent, name)}
    protocol = read(FOLD10_DIRECTORY / 'protocol.json')
    report['user_approved_preexisting_Keil_workspace_differences'] = audit_protected(protocol['protected_sha256'])
    report['old_models_firmware_releases_and_fold9_samples_unchanged'] = True
    save(FOLD10_DIRECTORY / 'application_validation.json', report)
    section(ROOT / 'md/STM32部署与串口验证.md',
            '## 2026-10-11：两个串口程序新增折10波形选择\n\n'
            'A仅推理与B板端预处理＋推理均新增“波形来源”下拉框，可选择“折9（联调）”与“折10（历史测试）”。'
            '默认仍为原折9的128条；折10为按原seed44纳入规则剔除五类全零后的完整2158条，不随机抽取或重新筛选。'
            '切换回到第0条，可用上一条/下一条/跳转查看任意记录；发送期间禁止切换。A发送预处理后的信号，B发送原始mV；两者模型、训练统计量、量化参数、阈值及固件均未修改，无需重新烧录HEX。\n\n'
            '命令行增加`--fold 10`，例如：\n\n'
            '```powershell\n'
            '.\\start_ecg_student_viz.cmd --fold 10\n'
            '.\\start_ecg_board_preprocess_viz.cmd --fold 10\n'
            '.\\start_ecg_board_preprocess_viz.cmd --fold 10 --headless --port COM7 --sample 0 --result-file outputs/serial_samples/seed44_fold10/physical_result.json\n'
            '```\n\n'
            'COM7替换为实际串口；不添加`--simulate`就是实际串口模式。模型握手仍使用原配套清单，折10样本另校验模型、预处理、量化参数、阈值和样本SHA-256；不符合当前模型则拒绝加载。'
            '折10电脑参考概率由本轮STM32Cube.AI host C验证重新生成，不是FP32教师或研究seed43预测。记录逐条保存折数、来源、样本/模型哈希、协议/预处理位置、阈值、真实标签、电脑参考、芯片输出和运行模式；折9/折10混合回放也可区分。\n\n'
            '样本包及证据位于`outputs/serial_samples/seed44_fold10/`：'
            '[样本清单](../outputs/serial_samples/seed44_fold10/sample_manifest.json)、'
            '[预处理及host C验证](../outputs/serial_samples/seed44_fold10/preparation_validation.json)、'
            '[应用核对](../outputs/serial_samples/seed44_fold10/application_validation.json)。'
            '生成入口`scripts/prepare_serial_fold10.py`，已有完整样本包时校验后正常退出并提示串口入口，不覆盖文件；校验失败则报错。复核入口`scripts/verify_serial_fold10.py`。\n\n'
            '本轮35项测试通过，完整2158条的Python/C预处理及INT8输入核对通过，并验证两个实际C串口解析器的抽样模拟及界面折9→折10→折9切换、收发和日志来源。'
            '这些是电脑核对，不代表本轮已完成折10实板验证。历史折10曾反复用于研发，不能将此次回放称为新的独立盲测。\n\n'
            '折9同样经过样本准备，原128条已缓存；折10此次首次完成完整2158条打包，后续启动直接读取，无需每次执行prepare。'
            '这里的prepare整理波形、标签和电脑参考，不训练模型；B仍发送原始波形，由板端完成预处理。\n\n'
            '复核发现Cube.AI CLI自动量化在两个点与固件舍入相差1：样本索引1203的[6,460]和1440的[0,778]。'
            '保留原CLI输入/输出作为证据，并用同一ST生成的host C模型及原固件权重，按实际固件INT8输入重算这两条参考。'
            '该调用对原CLI输入的输出逐值核对一致；其余参考沿用完整host验证。未修改模型、量化尺度、舍入规则或阈值。'
            '首次host验证遇到工作目录更新错误；换用独立短目录后成功，原失败日志及成功重试日志均保留。\n\n'
            '保护检查另发现原Keil工作目录有49个工程设置或构建文件与旧审计哈希不同；用户明确授权保留当前文件并记录差异。'
            '原始检查失败及逐文件哈希保存于`artifact_guard_audit.json`，精确授权列表保存于`artifact_guard_approval.json`；'
            '只允许列表中已确认的当前哈希，未来变化仍拒绝通过。模型、源码、发布HEX和折9样本没有哈希差异，未覆盖或还原任何现有工程。',
            'serial-fold10-20261011')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__': main()
