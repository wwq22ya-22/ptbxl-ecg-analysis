"""Label and signal helpers for the Chapman-Shaoxing/Ningbo ECG cohort.

The SNOMED-to-superclass map is deliberately conservative. Rhythm-only codes do
not establish the absence of morphological diagnoses, and unsupported
morphology codes make a record ineligible for the mapped-class metrics.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from src.data.ptbxl import LEAD_NAMES, SUPERCLASSES


DATASET_URL = "https://physionet.org/content/ecg-arrhythmia/1.0.0/"
ARCHIVE_URL = "https://physionet.org/content/ecg-arrhythmia/get-zip/1.0.0/"
SOURCE_FS_HZ = 500
TARGET_FS_HZ = 100

# This database has no explicit normal-ECG diagnosis code. Sinus rhythm alone
# is not treated as NORM because it is a rhythm label, not a morphology read.
RHYTHM_ONLY_CODES = frozenset(
    {
        "ABI", "APB", "AFIB", "AF", "AT", "AVNRT", "AVRT", "JEB", "JPT",
        "SA", "SAAWR", "SB", "SR", "ST", "SVT", "VB", "VEB", "VET", "VFW", "VPB",
        "WAVN",
    }
)

# Conservative hand-reviewed mapping from the dataset's official abbreviation
# dictionary to the PTB-XL diagnostic superclasses. Ambiguous codes are omitted
# on purpose and cause record exclusion rather than guessed labels.
CODE_TO_SUPERCLASS = {
    "MI": "MI",
    "MIBW": "MI",
    "MIFW": "MI",
    "MILW": "MI",
    "MISW": "MI",
    "STDD": "STTC",
    "STE": "STTC",
    "STTC": "STTC",
    "STTU": "STTC",
    "TWC": "STTC",
    "TWO": "STTC",
    "ERV": "STTC",
    "QTIE": "STTC",
    "1AVB": "CD",
    "2AVB": "CD",
    "2AVB1": "CD",
    "2AVB2": "CD",
    "3AVB": "CD",
    "AVB": "CD",
    "IDC": "CD",
    "IVB": "CD",
    "LBBB": "CD",
    "LBBBB": "CD",
    "LFBBB": "CD",
    "PRIE": "CD",
    "RBBB": "CD",
    "VPE": "CD",
    "WPW": "CD",
    "LVH": "HYP",
    "RAH": "HYP",
    "RVH": "HYP",
}

EVALUATED_CLASSES = ("MI", "STTC", "CD", "HYP")


def load_snomed_dictionary(path: str | Path) -> dict[str, str]:
    """Return SNOMED CT code -> dataset abbreviation from its official CSV."""
    result: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Acronym Name", "Snomed_CT"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected Chapman code dictionary columns in {path}.")
        for row in reader:
            code = str(row["Snomed_CT"]).strip()
            acronym = str(row["Acronym Name"]).strip().upper()
            if not code or not acronym:
                continue
            previous = result.setdefault(code, acronym)
            if previous != acronym:
                # Duplicate SNOMED concepts can have alternate abbreviations;
                # preserve both so the corresponding map resolves consistently.
                result[code] = previous + "|" + acronym
    return result


def parse_diagnosis_codes(comments: Iterable[str]) -> tuple[str, ...]:
    """Extract SNOMED CT diagnosis identifiers from WFDB header comments."""
    for comment in comments:
        match = re.match(r"\s*Dx\s*:\s*(.*)$", str(comment), flags=re.IGNORECASE)
        if match:
            return tuple(dict.fromkeys(re.findall(r"\d+", match.group(1))))
    return ()


def encode_mapped_labels(
    diagnosis_codes: Iterable[str],
    snomed_dictionary: dict[str, str],
) -> tuple[np.ndarray, bool, tuple[str, ...], tuple[str, ...]]:
    """Map complete supported morphologic diagnoses; return target and audit.

    The mask is true only when at least one morphology diagnosis is present and
    every non-rhythm diagnosis code can be mapped. Rhythm-only records and
    records carrying unmapped diagnoses are excluded from superclass metrics.
    """
    labels = np.zeros(len(SUPERCLASSES), dtype=np.uint8)
    class_positions = {name: index for index, name in enumerate(SUPERCLASSES)}
    seen_morphology = False
    unknown_codes: list[str] = []
    acronyms: list[str] = []
    for raw_code in diagnosis_codes:
        code = str(raw_code).strip()
        acronym_value = snomed_dictionary.get(code)
        if acronym_value is None:
            unknown_codes.append(code)
            seen_morphology = True
            continue
        for acronym in acronym_value.split("|"):
            acronyms.append(acronym)
            if acronym in RHYTHM_ONLY_CODES:
                continue
            seen_morphology = True
            superclass = CODE_TO_SUPERCLASS.get(acronym)
            if superclass is None:
                unknown_codes.append(f"{code}:{acronym}")
            else:
                labels[class_positions[superclass]] = 1
    complete = seen_morphology and not unknown_codes
    return labels, complete, tuple(acronyms), tuple(unknown_codes)


def convert_to_mv(signal: np.ndarray, units: Iterable[str]) -> np.ndarray:
    """Convert WFDB physical signals to mV, rejecting ambiguous units."""
    values = np.asarray(signal, dtype=np.float32)
    unit_list = tuple(str(unit).strip().replace("μ", "u").replace("µ", "u").casefold() for unit in units)
    if values.ndim != 2 or values.shape[1] != len(LEAD_NAMES) or len(unit_list) != len(LEAD_NAMES):
        raise ValueError("Expected a samples-by-12-leads signal and twelve unit names.")
    scales: list[float] = []
    for unit in unit_list:
        if unit in {"mv", "millivolt", "millivolts"}:
            scales.append(1.0)
        elif unit in {"uv", "microvolt", "microvolts"}:
            scales.append(0.001)
        elif unit in {"v", "volt", "volts"}:
            scales.append(1000.0)
        else:
            raise ValueError(f"Unsupported WFDB physical unit: {unit!r}.")
    converted = values * np.asarray(scales, dtype=np.float32)[None, :]
    if not np.isfinite(converted).all():
        raise ValueError("ECG contains NaN or infinite samples.")
    return converted


def resample_500_to_100(signal: np.ndarray) -> np.ndarray:
    """Anti-alias and decimate a ten-second 500 Hz ECG to model input rate."""
    from scipy.signal import resample_poly

    values = np.asarray(signal, dtype=np.float32)
    expected_shape = (10 * SOURCE_FS_HZ, len(LEAD_NAMES))
    if values.shape != expected_shape:
        raise ValueError(f"Expected a 10-second {SOURCE_FS_HZ} Hz ECG with shape {expected_shape}, got {values.shape}.")
    result = resample_poly(values, up=1, down=SOURCE_FS_HZ // TARGET_FS_HZ, axis=0)
    expected_output = (10 * TARGET_FS_HZ, len(LEAD_NAMES))
    if result.shape != expected_output:
        raise ValueError(f"Resampling produced {result.shape}; expected {expected_output}.")
    return np.asarray(result, dtype=np.float32)


