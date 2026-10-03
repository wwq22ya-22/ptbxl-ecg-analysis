"""Freeze the Cube.AI deployment contract and prepare validation waveforms."""
from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest
from src.data.ptbxl import load_ecg
from src.inference import load_predictor
from src.training.data import load_labeled_splits
from src.config import resolve_project_path


def write_spl_project(reference, destination, export, sdk, project_name="ECG_ResNet_INT8"):
    """Copy only source files; preserve the user's reference project unchanged."""
    if destination.exists():
        raise FileExistsError(f"Project already exists: {destination}")
    shutil.copytree(reference / "Lib", destination / "Lib")
    # Full-lib strtof alone has a >1.5 KiB call chain; reserve room for
    # AI callbacks and interrupts as well, rather than the reference's 1 KiB.
    startup = destination / "Lib/starup/startup_stm32f40_41xxx.s"
    content, replacements = re.subn(
        rb"(Stack_Size\s+EQU\s+)0x[0-9A-Fa-f]+",
        rb"\g<1>0x00002000", startup.read_bytes(), count=1,
    )
    if replacements != 1:
        raise ValueError("Reference startup stack definition needs manual integration.")
    startup.write_bytes(content)
    user = destination / "User"
    user.mkdir()
    for source in (reference / "User").iterdir():
        if source.is_file(): shutil.copy2(source, user / source.name)
    for name in ("usart.c", "usart.h"):
        path = user / name
        # Keep the reference's GBK comments intact while fixing ISR visibility.
        content = path.read_bytes().replace(b"u16 USART_RX_STA", b"volatile u16 USART_RX_STA")
        path.write_bytes(content)
    interrupt = user / "stm32f4xx_it.c"
    content = interrupt.read_bytes()
    content, replacements = re.subn(rb"void SysTick_Handler\(void\)\s*\{\s*\}",
                                    b"void SysTick_Handler(void)\n{\n    ecg_tick_1ms();\n}", content)
    if replacements != 1: raise ValueError("Reference SysTick handler needs manual integration.")
    interrupt.write_bytes(b'#include "ecg_tick.h"\n' + content)
    for name in ["ecg_protocol.c", "ecg_protocol.h", "ecg_tick.h"]:
        shutil.copy2(ROOT / "firmware/ecg_serial" / name, user / name)
    header = export / "ecg_deployment_config.h"
    if not header.exists():
        header = ROOT / "firmware/ecg_serial/ecg_deployment_config.h"
    shutil.copy2(header, user / "ecg_deployment_config.h")
    shutil.copy2(ROOT / "firmware/ecg_serial/main_spl.c", user / "main.c")
    ai = user / "ai"
    ai.mkdir()
    for source in (export / "generated").iterdir():
        if source.suffix in (".c", ".h") or source.name == "LICENSE.txt":
            shutil.copy2(source, ai / source.name)
    shutil.copytree(sdk / "Inc", ai / "st_ai/Inc")
    runtime = sdk / "Lib/MDK/ARMCortexM4/NetworkRuntime1020_CM4_Keil.lib"
    (destination / "Lib/AI").mkdir(exist_ok=True)
    shutil.copy2(runtime, destination / "Lib/AI" / runtime.name)
    project = ET.parse(reference / "MDK_Project/Project.uvprojx")
    root = project.getroot()
    groups = root.find(".//Groups")
    for group in list(groups):
        if group.findtext("GroupName") in {"AI", "ST_AI", "Text"}: groups.remove(group)
    group = ET.SubElement(groups, "Group")
    ET.SubElement(group, "GroupName").text = "ECG AI"
    files = ET.SubElement(group, "Files")
    for name, path in [("ecg_protocol.c", "..\\User\\ecg_protocol.c")] + [
        (source.name, "..\\User\\ai\\" + source.name) for source in ai.glob("*.c")
    ]:
        item = ET.SubElement(files, "File")
        ET.SubElement(item, "FileName").text = name
        ET.SubElement(item, "FileType").text = "1"
        ET.SubElement(item, "FilePath").text = path
    # Enforce the smaller VE Flash limit until the physical chip is confirmed.
    root.find(".//Device").text = "STM32F407VETx"
    root.find(".//OutputName").text = project_name
    for node in root.iter():
        if node.text:
            node.text = node.text.replace("STM32F407VGTx", "STM32F407VETx")
    driver = root.find(".//FlashDriverDll")
    driver.text = driver.text.replace("STM32F4xx_1024", "STM32F4xx_512").replace("-FL0100000", "-FL0080000")
    cpu = root.find(".//Cpu")
    cpu.text = cpu.text.replace("0x00100000", "0x00080000")
    for node in root.findall(".//IROM/Size") + root.findall(".//OCR_RVCT1/Size"):
        node.text = "0x80000"
    for node in root.findall(".//OnChipMemories//Size"):
        if node.text == "0x100000": node.text = "0x80000"
    root.find(".//pCCUsed").text = "5060422::V5.06 update 4 (build 422)::ARMCC"
    root.find(".//Optim").text = "2"
    # printf float support and strtof/nearbyintf use the full Arm C library.
    root.find(".//useUlib").text = "0"
    target_dir = destination / "MDK_Project"
    target_dir.mkdir()
    target = target_dir / f"{project_name}.uvprojx"
    ET.indent(project)
    project.write(target, encoding="UTF-8", xml_declaration=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=ROOT / "outputs/cubeai/resnet")
    parser.add_argument("--reference-mdk", type=Path)
    parser.add_argument("--sdk", type=Path)
    parser.add_argument("--project-dir", type=Path, default=ROOT / "outputs/cubeai/stm32_spl_project")
    parser.add_argument("--project-name", default="ECG_ResNet_INT8")
    args = parser.parse_args()
    export = args.export_dir.resolve()
    base = json.loads((export / "deployment_manifest.json").read_text(encoding="utf-8"))
    quant = json.loads((export / "int8_validation_report.json").read_text(encoding="utf-8"))
    model = export / quant["onnx"]["file"]
    if digest(model) != quant["onnx"]["sha256"]: raise ValueError("INT8 artifact changed.")
    cinfo = json.loads((export / "generated/ecg_network_c_info.json").read_text())
    graph = cinfo["graphs"][0]
    buffers = {buffer["id"]: buffer for buffer in cinfo["buffers"]}
    tensor = buffers[graph["inputs"][0]]
    output = buffers[graph["outputs"][0]]
    if (tensor["format"] != "STAI_FORMAT_S8" or tensor["shape"] != [12, 1000]
            or not tensor["channel_first"] or output["format"] != "STAI_FORMAT_FLOAT"):
        raise ValueError("Generated C input/output contract differs from serial firmware.")
    input_scale = tensor["intq"]["scales"][0]
    input_zero = tensor["intq"]["offsets"][0]
    checkpoint = torch.load(base["source_checkpoint"], map_location="cpu", weights_only=True)
    predictor = load_predictor(base["source_checkpoint"], device="cpu")
    dataset = resolve_project_path(checkpoint["config"]["dataset"]["root"], ROOT)
    frame = load_labeled_splits(dataset, 100, require_diagnostic_label=bool(
        checkpoint['config']['dataset'].get('require_diagnostic_label', False)
    ))["validation"].set_index("ecg_id")
    source = np.load(export / "validation_reference.npz")
    raw = []
    for i, ecg_id in enumerate(source["ecg_id"]):
        record = load_ecg(dataset, frame.loc[int(ecg_id), "filename_lr"], sampling_rate_hz=100)
        prepared = predictor.prepare_tensor(record.signal, sampling_rate_hz=100,
                                            lead_names=record.lead_names, units=record.units).cpu().numpy()[0]
        np.testing.assert_allclose(prepared, source["ecg"][i], atol=1e-6, rtol=1e-6)
        raw.append(record.signal.T)
    cpu_c = np.load(export / "host_validation/ecg_network_val_io.npz")["c_outputs_1"].reshape(-1, 5)
    cpu_p = 1 / (1 + np.exp(-np.clip(cpu_c, -80, 80)))
    samples = export / "serial_samples.npz"
    np.savez_compressed(samples, raw_mv=np.stack(raw), ecg=source["ecg"],
                        targets=source["targets"], ecg_id=source["ecg_id"],
                        reference_probabilities=cpu_p)
    manifest = {
        "protocol": 1, "model_type": base["model_type"], "model_sha256": digest(model),
        "model_file": model.name, "sample_file": samples.name, "sample_sha256": digest(samples),
        "sample_source": "PTB-XL validation fold 9", "sample_count": len(raw),
        "reference": "Cube.AI host C-model; not physical-board outputs",
        "labels": base["outputs"]["labels"], "thresholds": quant["thresholds"],
        "lead_order": base["signal_contract"]["lead_order"], "input_shape": [12, 1000],
        "input_scale": input_scale, "input_zero_point": input_zero,
        "preprocessing": base["signal_contract"]["preprocessing"], "preprocessing_location": "host",
        "default_baudrate": 115200, "chunk_values": 64,
    }
    (export / "serial_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    header = (
        "/* Generated from the frozen Cube.AI C input contract. */\n"
        "#ifndef ECG_DEPLOYMENT_CONFIG_H\n#define ECG_DEPLOYMENT_CONFIG_H\n"
        f'#define ECG_MODEL_SHA256 "{manifest["model_sha256"]}"\n'
        f"#define ECG_INPUT_SCALE {input_scale:.17g}f\n#define ECG_INPUT_ZERO_POINT ({input_zero})\n"
        "static const float ECG_THRESHOLDS[5] = { "
        + ", ".join(f"{value:.9f}f" for value in manifest["thresholds"]) + " };\n#endif\n"
    )
    (export / "ecg_deployment_config.h").write_text(header, encoding="utf-8")
    print(f"Serial samples: {samples}; raw-vs-cache preprocessing parity passed for {len(raw)} records.")
    if args.reference_mdk:
        if not args.sdk: raise ValueError("--sdk is required for the Keil project.")
        target = write_spl_project(args.reference_mdk, args.project_dir.resolve(), export, args.sdk, args.project_name)
        print(f"Keil project: {target}")


if __name__ == "__main__":
    main()
