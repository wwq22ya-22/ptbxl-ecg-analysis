"""ECG waveform -> serial -> STM32 five-label classification viewer."""
from __future__ import annotations
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest
from src.inference.serial_link import STM32Link, load_serial_manifest


def read_samples(manifest, directory):
    path = directory / manifest["sample_file"]
    if digest(path) != manifest["sample_sha256"]: raise ValueError("验证波形包已变化，请重新生成部署包。")
    with np.load(path, allow_pickle=False) as data:
        samples = {name: data[name].copy() for name in data.files}
    n = len(samples["ecg_id"])
    for key in ("ecg", "raw_mv"):
        if samples[key].shape != (n, 12, 1000) or not np.isfinite(samples[key]).all():
            raise ValueError(f"波形 {key} 形状或数值不正确。")
    if samples["targets"].shape != (n, 5) or not np.isin(samples["targets"], [0, 1]).all():
        raise ValueError("标签数组无效。")
    return samples


def transport_for(args, manifest, directory):
    if args.simulate:
        from src.inference.serial_simulator import onnx_simulator
        return onnx_simulator(manifest, directory)
    import serial
    if not args.port: raise ValueError("请指定 --port COM端口，或使用 --simulate 模拟。")
    return serial.Serial(args.port, args.baudrate, timeout=0.2, write_timeout=3,
                         bytesize=8, parity="N", stopbits=1)


def headless(args, manifest, samples, directory):
    if not 0 <= args.sample < len(samples["ecg_id"]): raise ValueError("样本索引超出范围。")
    transport = transport_for(args, manifest, directory)
    try:
        link = STM32Link(transport, manifest)
        link.handshake()
        start = time.monotonic()
        result = link.predict(samples["ecg"][args.sample])
        result.update(mode="simulation" if args.simulate else "physical_serial",
                      ecg_id=int(samples["ecg_id"][args.sample]),
                      total_seconds=time.monotonic() - start,
                      truth=[name for name, hit in zip(manifest["labels"], samples["targets"][args.sample]) if hit])
        result["max_probability_difference_from_host_c"] = float(np.max(np.abs(
            np.asarray(result["probabilities"]) - samples["reference_probabilities"][args.sample])))
        if args.result_file:
            args.result_file.parent.mkdir(parents=True, exist_ok=True)
            args.result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        transport.close()


def launch_gui(args, manifest, samples, directory):
    import tkinter as tk
    from tkinter import ttk, messagebox
    import serial.tools.list_ports
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    class App:
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("ECG 五类 STM32 串口验证")
            self.root.geometry("1120x830")
            self.messages = queue.Queue()
            self.transport = None
            self.link = None
            self.busy = False
            self.closing = False
            self.index = args.sample
            self.results = []
            self.mode = tk.StringVar(value="模拟验证" if args.simulate else "真实串口")
            self.port = tk.StringVar(value=args.port or "")
            self.baud = tk.StringVar(value=str(args.baudrate))
            self.lead = tk.StringVar(value=manifest["lead_order"][0])
            self.jump = tk.StringVar(value=str(self.index))
            self.status = tk.StringVar(value="请选择串口并连接；也可选择模拟验证。")
            outer = ttk.Frame(self.root, padding=14)
            outer.pack(fill="both", expand=True)
            controls = ttk.Frame(outer); controls.pack(fill="x")
            ttk.Label(controls, text="模式").pack(side="left")
            self.mode_box = ttk.Combobox(controls, textvariable=self.mode,
                                        values=["真实串口", "模拟验证"], state="readonly", width=10)
            self.mode_box.pack(side="left", padx=6)
            self.port_box = ttk.Combobox(controls, textvariable=self.port, width=10)
            self.port_box.pack(side="left", padx=6)
            self.refresh_button = ttk.Button(controls, text="刷新串口", command=self.refresh_ports)
            self.refresh_button.pack(side="left", padx=6)
            ttk.Label(controls, text="波特率").pack(side="left")
            self.baud_box = ttk.Combobox(controls, textvariable=self.baud,
                                        values=[115200, 230400, 460800, 921600], width=9, state="readonly")
            self.baud_box.pack(side="left", padx=6)
            self.connect_button = ttk.Button(controls, text="连接", command=self.connect)
            self.connect_button.pack(side="left", padx=6)
            ttk.Label(outer, text="12 导联 · 100 Hz · 10 秒 · NORM / MI / STTC / CD / HYP",
                      font=("Microsoft YaHei", 13, "bold")).pack(anchor="w", pady=(16, 6))
            self.record_label = ttk.Label(outer, text=""); self.record_label.pack(anchor="w")
            lead_controls = ttk.Frame(outer); lead_controls.pack(fill="x", pady=8)
            ttk.Label(lead_controls, text="显示导联").pack(side="left")
            selector = ttk.Combobox(lead_controls, textvariable=self.lead,
                                    values=manifest["lead_order"], state="readonly", width=8)
            selector.pack(side="left", padx=6)
            selector.bind("<<ComboboxSelected>>", lambda _: self.draw_waveform())
            ttk.Label(lead_controls, text="图中显示原始 mV 波形；发送的是训练流程预处理后的 12 导联信号。"
                      ).pack(side="left", padx=8)
            self.figure = Figure(figsize=(10, 3), dpi=100)
            self.axis = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, master=outer)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.table = ttk.Treeview(outer, columns=("truth", "reference", "chip", "threshold", "result"),
                                      show="tree headings", height=5)
            self.table.heading("#0", text="类别"); self.table.column("#0", width=100)
            for column, text in zip(self.table["columns"], ["真实标签", "电脑 C 模型参考", "芯片返回概率", "判定阈值", "芯片判断"]):
                self.table.heading(column, text=text); self.table.column(column, width=160, anchor="center")
            for name in manifest["labels"]: self.table.insert("", "end", iid=name, text=name)
            self.table.pack(fill="x", pady=10)
            self.result_label = ttk.Label(outer, text="芯片结果：等待发送", font=("Microsoft YaHei", 12, "bold"))
            self.result_label.pack(anchor="w", pady=4)
            self.progress = ttk.Progressbar(outer, maximum=100); self.progress.pack(fill="x", pady=6)
            navigation = ttk.Frame(outer); navigation.pack(fill="x", pady=8)
            self.previous = ttk.Button(navigation, text="上一条", command=lambda: self.change(self.index - 1))
            self.previous.pack(side="left")
            self.index_entry = ttk.Entry(navigation, textvariable=self.jump, width=7)
            self.index_entry.pack(side="left", padx=6)
            self.index_entry.bind("<Return>", lambda _: self.jump_to())
            self.jump_button = ttk.Button(navigation, text="跳转", command=self.jump_to); self.jump_button.pack(side="left")
            self.next = ttk.Button(navigation, text="下一条", command=lambda: self.change(self.index + 1))
            self.next.pack(side="left", padx=6)
            self.send_button = ttk.Button(navigation, text="发送波形并分类", command=self.send, state="disabled")
            self.send_button.pack(side="right")
            ttk.Button(navigation, text="保存验证记录", command=self.save_results).pack(side="right", padx=8)
            ttk.Label(outer, textvariable=self.status, wraplength=1050).pack(anchor="w", pady=6)
            self.root.protocol("WM_DELETE_WINDOW", self.close)
            self.refresh_ports()
            self.change(self.index)
            self.root.after(50, self.poll)

        def refresh_ports(self):
            ports = [port.device for port in serial.tools.list_ports.comports()]
            self.port_box["values"] = ports
            if not self.port.get() and ports: self.port.set(ports[0])

        def draw_waveform(self):
            lead = manifest["lead_order"].index(self.lead.get())
            self.axis.clear()
            self.axis.plot(np.arange(1000) / 100, samples["raw_mv"][self.index, lead], color="#176b89", linewidth=1)
            self.axis.set(xlabel="时间 / 秒", ylabel="电压 / mV", title=f"ECG {samples['ecg_id'][self.index]} · {self.lead.get()}")
            self.axis.grid(alpha=0.2)
            self.figure.tight_layout()
            self.canvas.draw_idle()

        def change(self, index):
            if self.busy or not 0 <= index < len(samples["ecg_id"]): return
            self.index = index; self.jump.set(str(index))
            truth = [name for name, hit in zip(manifest["labels"], samples["targets"][index]) if hit]
            label_text = '、'.join(truth) if truth else '未标注这五类诊断标签（不代表正常）'
            self.record_label["text"] = f"样本 {index} / {len(samples['ecg_id'])-1}　记录 {samples['ecg_id'][index]}　真实标签：{label_text}　来源：验证折 9"
            for i, name in enumerate(manifest["labels"]):
                target_text = ('阳性' if samples['targets'][index, i] else '阴性') if truth else '未标注'
                self.table.item(name, values=(target_text,
                    f"{samples['reference_probabilities'][index,i]:.2%}", "—", f"{manifest['thresholds'][i]:.3f}", "—"))
            self.result_label["text"] = "芯片结果：等待发送"
            self.progress["value"] = 0
            self.draw_waveform()

        def jump_to(self):
            try:
                value = int(self.jump.get())
                if not 0 <= value < len(samples["ecg_id"]): raise ValueError()
                self.change(value)
            except ValueError: messagebox.showerror("索引无效", f"请输入 0 到 {len(samples['ecg_id'])-1}。")

        def set_busy(self, busy):
            self.busy = busy
            for widget in (self.previous, self.next, self.jump_button, self.index_entry, self.connect_button):
                widget["state"] = "disabled" if busy else "normal"
            self.send_button["state"] = "normal" if self.link and not busy else "disabled"
            connected = self.transport is not None
            for widget in (self.mode_box, self.port_box, self.baud_box):
                widget["state"] = "disabled" if busy or connected else "readonly"
            self.refresh_button["state"] = "disabled" if busy or connected else "normal"

        def connect(self):
            if self.transport:
                self.transport.close(); self.transport = None; self.link = None
                self.connect_button["text"] = "连接"
                self.status.set("串口已关闭。"); self.set_busy(False); return
            mode, port, baud = self.mode.get(), self.port.get(), self.baud.get()
            self.set_busy(True); self.status.set("正在连接并核对芯片模型…")
            def work():
                transport = None
                try:
                    from argparse import Namespace
                    config = Namespace(simulate=mode == "模拟验证", port=port, baudrate=int(baud))
                    transport = transport_for(config, manifest, directory)
                    link = STM32Link(transport, manifest); link.handshake()
                    if self.closing: transport.close(); return
                    self.messages.put(("connected", (transport, link, mode)))
                except Exception as error:
                    if transport: transport.close()
                    self.messages.put(("error", str(error)))
            threading.Thread(target=work, daemon=True).start()

        def send(self):
            if not self.link or self.busy: return
            index = self.index; mode = self.mode.get(); link = self.link
            self.set_busy(True); self.status.set("正在分块发送波形，完成后等待芯片分类…")
            self.result_label["text"] = "等待分类结果…"
            def work():
                try:
                    start = time.monotonic()
                    result = link.predict(samples["ecg"][index], progress=lambda x: self.messages.put(("progress", x)))
                    result.update(ecg_id=int(samples["ecg_id"][index]), sample_index=index,
                                  mode="simulation" if mode == "模拟验证" else "physical_serial",
                                  total_seconds=time.monotonic() - start,
                                  truth=[name for name, hit in zip(manifest["labels"], samples["targets"][index]) if hit],
                                  host_c_reference_probabilities=samples["reference_probabilities"][index].tolist(),
                                  max_probability_difference_from_host_c=float(np.max(np.abs(
                                      np.asarray(result["probabilities"]) - samples["reference_probabilities"][index]))))
                    self.messages.put(("result", result))
                except Exception as error: self.messages.put(("error", str(error)))
            threading.Thread(target=work, daemon=True).start()

        def poll(self):
            if self.closing: return
            while not self.messages.empty():
                kind, value = self.messages.get_nowait()
                if kind == "connected":
                    self.transport, self.link, mode = value
                    self.connect_button["text"] = "断开"
                    self.status.set("模型核对成功。" + (" 当前为模拟模式，结果和耗时不代表实板。" if mode == "模拟验证" else "可以发送波形。"))
                    self.set_busy(False)
                elif kind == "progress": self.progress["value"] = value * 100
                elif kind == "error":
                    self.status.set(value); self.set_busy(False)
                    self.result_label["text"] = "操作失败，请查看下方提示"
                elif kind == "result":
                    self.results.append(value)
                    for i, name in enumerate(manifest["labels"]):
                        row = list(self.table.item(name, "values"))
                        row[2] = f"{value['probabilities'][i]:.2%}"
                        row[4] = "阳性" if value["mask"] & (1 << i) else "阴性"
                        self.table.item(name, values=row)
                    labels = "、".join(value["positive_labels"]) or "无标签达到阈值"
                    simulated = value["mode"] == "simulation"
                    self.result_label["text"] = ("模拟结果：" if simulated else "芯片结果：") + labels
                    difference = np.max(np.abs(np.asarray(value["probabilities"]) - samples["reference_probabilities"][value["sample_index"]]))
                    timing = "模拟模式不测量芯片耗时" if simulated else f"芯片推理 {value['inference_ms']} ms"
                    self.status.set(f"{timing}；收发总耗时 {value['total_seconds']:.2f} s；与电脑 C 模型最大概率差 {difference:.6f}。")
                    self.set_busy(False)
            self.root.after(50, self.poll)

        def save_results(self):
            if not self.results: messagebox.showinfo("暂无记录", "先发送一条波形完成验证。"); return
            path = ROOT / "outputs/cubeai/serial_validation_log.json"
            path.write_text(json.dumps({"model_sha256": manifest["model_sha256"], "results": self.results},
                                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.status.set(f"验证记录已保存：{path}")

        def close(self):
            self.closing = True
            if self.transport: self.transport.close()
            self.root.destroy()

    window = App()
    if args.smoke_test:
        window.root.update()
        window.canvas.draw()
        window.figure.savefig(ROOT / "outputs/cubeai/serial_waveform_preview.png")
        window.close()
        print("GUI initialization and waveform rendering passed.")
    else: window.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/cubeai/resnet/serial_manifest.json")
    parser.add_argument("--port")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    manifest = load_serial_manifest(args.manifest)
    directory = args.manifest.resolve().parent
    samples = read_samples(manifest, directory)
    if not 0 <= args.sample < len(samples["ecg_id"]): raise ValueError("样本索引超出范围。")
    if args.headless: headless(args, manifest, samples, directory)
    else: launch_gui(args, manifest, samples, directory)


if __name__ == "__main__":
    main()
