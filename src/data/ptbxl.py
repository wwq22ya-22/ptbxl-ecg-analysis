"""PTB-XL path discovery, metadata/label handling, and WFDB waveform loading."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PTBXL_URL = "https://physionet.org/content/ptb-xl/1.0.3/"
LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
_METADATA_FILE = "ptbxl_database.csv"
_STATEMENTS_FILE = "scp_statements.csv"


class PTBXLDataError(RuntimeError):
    """Raised when PTB-XL files or metadata are missing or inconsistent."""


@dataclass(frozen=True)
class ECGRecord:
    """A WFDB record in samples-by-leads layout with physical units."""

    signal: np.ndarray
    sampling_rate_hz: int
    lead_names: tuple[str, ...]
    units: tuple[str, ...]
    record_name: str


def _dataset_requirements(sampling_rate_hz: int) -> tuple[str, ...]:
    if sampling_rate_hz == 100:
        return (_METADATA_FILE, _STATEMENTS_FILE, "records100/")
    if sampling_rate_hz == 500:
        return (_METADATA_FILE, _STATEMENTS_FILE, "records500/")
    raise ValueError("sampling_rate_hz must be 100 or 500 for PTB-XL.")


def _looks_like_dataset(path: Path, sampling_rate_hz: int) -> bool:
    records_dir = "records100" if sampling_rate_hz == 100 else "records500"
    return (
        (path / _METADATA_FILE).is_file()
        and (path / _STATEMENTS_FILE).is_file()
        and (path / records_dir).is_dir()
    )


def _candidate_directories(base: Path) -> list[Path]:
    """Return likely dataset roots without traversing the waveform tree."""
    base = base.expanduser().resolve()
    candidates = [base, base / "ptb-xl", base / "ptbxl"]
    for parent in (base / "ptb-xl", base / "ptbxl", base):
        if parent.is_dir():
            try:
                children = sorted(parent.iterdir(), key=lambda p: p.name.casefold())
            except OSError:
                continue
            candidates.extend(child for child in children if child.is_dir())
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def find_ptbxl_root(
    dataset_root: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    sampling_rate_hz: int = 100,
) -> Path:
    """Find a complete PTB-XL folder at the supplied or common local paths.

    Searches only likely roots and their immediate children, so it never scans
    every waveform file. An explicit path may be either the dataset directory
    itself or a parent containing a standard PTB-XL directory.
    """
    _dataset_requirements(sampling_rate_hz)
    if dataset_root is not None:
        base = Path(dataset_root)
        bases = [base]
    else:
        project = Path(project_root or Path.cwd()).expanduser().resolve()
        bases = [project / "data" / "ptb-xl", project / "data" / "ptbxl", project / "data"]

    checked: list[Path] = []
    for base in bases:
        for candidate in _candidate_directories(base):
            checked.append(candidate)
            if _looks_like_dataset(candidate, sampling_rate_hz):
                return candidate.resolve()

    expected = "\n".join(f"  - {item}" for item in _dataset_requirements(sampling_rate_hz))
    checked_text = "\n".join(f"  - {path}" for path in checked[:12]) or "  - no candidate directories found"
    raise PTBXLDataError(
        "PTB-XL dataset was not found or is incomplete.\n"
        f"Official source: {PTBXL_URL}\n"
        f"Place the {sampling_rate_hz} Hz data under data/ptb-xl/ (or pass --dataset-root).\n"
        f"Required items:\n{expected}\n"
        f"Checked locations:\n{checked_text}"
    )


def load_ptbxl_metadata(
    dataset_root: str | Path,
    *,
    sampling_rate_hz: int = 100,
) -> pd.DataFrame:
    """Load metadata and verify the fields required by the official split."""
    root = find_ptbxl_root(dataset_root, sampling_rate_hz=sampling_rate_hz)
    metadata_path = root / _METADATA_FILE
    frame = pd.read_csv(metadata_path)
    filename_column = "filename_lr" if sampling_rate_hz == 100 else "filename_hr"
    required = {"ecg_id", "patient_id", "scp_codes", "strat_fold", filename_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PTBXLDataError(f"{metadata_path} is missing required columns: {', '.join(missing)}")
    frame["strat_fold"] = pd.to_numeric(frame["strat_fold"], errors="coerce").astype("Int64")
    if frame["strat_fold"].isna().any():
        rows = frame.index[frame["strat_fold"].isna()].tolist()[:5]
        raise PTBXLDataError(f"Invalid or missing strat_fold values at metadata rows: {rows}")
    invalid_folds = sorted(set(frame["strat_fold"].astype(int)) - set(range(1, 11)))
    if invalid_folds:
        raise PTBXLDataError(f"strat_fold values must be 1-10; found {invalid_folds}")
    return frame


def load_scp_statements(
    dataset_root: str | Path,
    *,
    sampling_rate_hz: int = 100,
) -> pd.DataFrame:
    """Load SCP-ECG mappings indexed by statement code."""
    root = find_ptbxl_root(dataset_root, sampling_rate_hz=sampling_rate_hz)
    path = root / _STATEMENTS_FILE
    statements = pd.read_csv(path, index_col=0)
    required = {"diagnostic", "diagnostic_class"}
    missing = sorted(required.difference(statements.columns))
    if missing:
        raise PTBXLDataError(f"{path} is missing required columns: {', '.join(missing)}")
    statements.index = statements.index.map(str)
    return statements


def parse_scp_codes(value: Any) -> dict[str, float]:
    """Safely parse the SCP-code dictionary stored in PTB-XL CSV metadata."""
    if value is None:
        return {}
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return {}
    except (TypeError, ValueError):
        pass
    parsed = value if isinstance(value, Mapping) else ast.literal_eval(str(value))
    if not isinstance(parsed, Mapping):
        raise ValueError(f"scp_codes must contain a mapping, got {type(parsed).__name__}.")
    result: dict[str, float] = {}
    for code, likelihood in parsed.items():
        try:
            result[str(code)] = float(likelihood)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid likelihood for SCP code {code!r}: {likelihood!r}") from exc
    return result


def _is_diagnostic(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def encode_superclass_labels(
    scp_codes: Any,
    statements: pd.DataFrame,
    *,
    superclasses: Sequence[str] = SUPERCLASSES,
) -> np.ndarray:
    """Map diagnostic SCP codes to a multi-hot superclass target vector.

    Unknown and non-diagnostic SCP codes are ignored. A target is positive when
    at least one present diagnostic code maps to that superclass.
    """
    codes = parse_scp_codes(scp_codes)
    targets = np.zeros(len(superclasses), dtype=np.float32)
    positions = {name: index for index, name in enumerate(superclasses)}
    for code in codes:
        # A likelihood of 0 in PTB-XL means unknown likelihood, not absence.
        if code not in statements.index:
            continue
        row = statements.loc[code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if not _is_diagnostic(row["diagnostic"]):
            continue
        diagnostic_class = row["diagnostic_class"]
        if pd.isna(diagnostic_class):
            continue
        position = positions.get(str(diagnostic_class).strip())
        if position is not None:
            targets[position] = 1.0
    return targets


def add_superclass_labels(
    metadata: pd.DataFrame,
    statements: pd.DataFrame,
    *,
    superclasses: Sequence[str] = SUPERCLASSES,
) -> pd.DataFrame:
    """Return a copy of metadata with one binary column per superclass."""
    if "scp_codes" not in metadata.columns:
        raise PTBXLDataError("Metadata has no scp_codes column.")
    result = metadata.copy()
    vectors = [
        encode_superclass_labels(value, statements, superclasses=superclasses)
        for value in result["scp_codes"]
    ]
    if vectors:
        matrix = np.stack(vectors)
    else:
        matrix = np.zeros((0, len(superclasses)), dtype=np.float32)
    for index, name in enumerate(superclasses):
        result[name] = matrix[:, index].astype(np.uint8)
    return result


def create_fold_splits(
    metadata: pd.DataFrame,
    *,
    train_folds: Sequence[int] = tuple(range(1, 9)),
    validation_fold: int = 9,
    test_fold: int = 10,
    test_folds: Sequence[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Create patient-safe train/validation/test subsets from official folds."""
    required = {"patient_id", "strat_fold"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise PTBXLDataError(f"Metadata is missing split columns: {', '.join(missing)}")
    folds = set(metadata["strat_fold"].astype(int).unique())
    evaluation_folds = tuple(test_folds) if test_folds is not None else (test_fold,)
    if not train_folds or not evaluation_folds:
        raise ValueError("Training and evaluation fold lists must not be empty.")
    requested = set(train_folds) | {validation_fold} | set(evaluation_folds)
    absent = sorted(requested - folds)
    if absent:
        raise PTBXLDataError(f"Requested strat_fold values are absent from metadata: {absent}")
    if set(train_folds) & ({validation_fold} | set(evaluation_folds)) or validation_fold in evaluation_folds:
        raise ValueError("Train, validation, and test fold assignments must be disjoint.")

    fold_values = metadata["strat_fold"].astype(int)
    split = {
        "train": metadata.loc[fold_values.isin(train_folds)].copy(),
        "validation": metadata.loc[fold_values.eq(validation_fold)].copy(),
        "test": metadata.loc[fold_values.isin(evaluation_folds)].copy(),
    }
    patient_sets = {name: set(part["patient_id"].dropna()) for name, part in split.items()}
    names = tuple(split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                examples = sorted(overlap, key=str)[:5]
                raise PTBXLDataError(
                    f"Patient leakage between {left} and {right} folds; examples: {examples}"
                )
    return split


def _record_base_path(dataset_root: Path, relative_name: str) -> Path:
    normalized = relative_name.replace("\\", os.sep).replace("/", os.sep)
    base = (dataset_root / normalized).resolve()
    try:
        base.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise PTBXLDataError(f"Record path escapes the dataset directory: {relative_name!r}") from exc
    if base.suffix.casefold() in {".hea", ".dat"}:
        base = base.with_suffix("")
    if not base.with_suffix(".hea").is_file():
        raise PTBXLDataError(f"WFDB header not found for record {relative_name!r}: {base.with_suffix('.hea')}")
    return base


def load_ecg(
    dataset_root: str | Path,
    record_name: str,
    *,
    sampling_rate_hz: int = 100,
) -> ECGRecord:
    """Load one WFDB record and reorder channels into the standard 12-lead order."""
    root = find_ptbxl_root(dataset_root, sampling_rate_hz=sampling_rate_hz)
    base_path = _record_base_path(root, record_name)
    try:
        import wfdb
    except ImportError as exc:
        raise PTBXLDataError("WFDB support is not installed. Install project dependencies with pip install -r requirements.txt.") from exc
    try:
        signal, fields = wfdb.rdsamp(str(base_path))
    except Exception as exc:
        raise PTBXLDataError(f"Could not read WFDB record {record_name!r}: {exc}") from exc

    signal = np.asarray(signal, dtype=np.float32)
    names = tuple(str(name).strip() for name in fields.get("sig_name", ()))
    raw_units = fields.get("units", fields.get("sig_units", ()))
    units = tuple(str(unit).strip() for unit in raw_units)
    actual_fs = int(round(float(fields.get("fs", sampling_rate_hz))))
    if signal.ndim != 2 or signal.shape[1] != len(names):
        raise PTBXLDataError(f"WFDB signal shape {signal.shape} does not match its lead names {names}.")
    if actual_fs != sampling_rate_hz:
        raise PTBXLDataError(
            f"Record sampling rate is {actual_fs} Hz, expected {sampling_rate_hz} Hz. "
            "Use the matching filename column and sampling-rate setting."
        )
    normalized_names = [name.casefold() for name in names]
    if len(set(normalized_names)) != len(normalized_names):
        raise PTBXLDataError(f"Duplicate lead names in record {record_name!r}: {names}")
    positions = {name: index for index, name in enumerate(normalized_names)}
    missing_leads = [lead for lead in LEAD_NAMES if lead.casefold() not in positions]
    if missing_leads:
        raise PTBXLDataError(f"Record {record_name!r} is missing standard leads: {missing_leads}")
    order = [positions[lead.casefold()] for lead in LEAD_NAMES]
    signal = signal[:, order]
    reordered_units = tuple(units[index] for index in order) if len(units) == len(names) else tuple("unknown" for _ in LEAD_NAMES)
    if not np.isfinite(signal).all():
        raise PTBXLDataError(f"Record {record_name!r} contains NaN or infinite samples.")
    return ECGRecord(
        signal=signal,
        sampling_rate_hz=actual_fs,
        lead_names=LEAD_NAMES,
        units=reordered_units,
        record_name=record_name,
    )
