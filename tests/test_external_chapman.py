"""External cohort data mapping and signal conversion tests."""

from __future__ import annotations

import numpy as np

from src.data.ptbxl import SUPERCLASSES
from src.external_validation.chapman import (
    EVALUATED_CLASSES,
    convert_to_mv,
    encode_mapped_labels,
    load_snomed_dictionary,
    parse_diagnosis_codes,
    resample_500_to_100,
)


def test_load_official_code_dictionary_and_parse_header(tmp_path):
    dictionary_file = tmp_path / "ConditionNames_SNOMED-CT.csv"
    dictionary_file.write_text(
        "Acronym Name,Full Name,Snomed_CT\n"
        "1AVB,1 degree atrioventricular block,270492004\n"
        "SR,Sinus Rhythm,426783006\n",
        encoding="utf-8-sig",
    )
    mapping = load_snomed_dictionary(dictionary_file)
    assert mapping["270492004"] == "1AVB"
    assert parse_diagnosis_codes(["Age: 62", "Dx: 270492004, 426783006"]) == (
        "270492004", "426783006"
    )


def test_labels_only_include_explicit_and_complete_morphology_codes():
    dictionary = {
        "426783006": "SR",
        "164865005": "MI",
        "164909002": "LBBB",
        "999999": "UNREVIEWED",
    }
    rhythm_labels, rhythm_complete, _, rhythm_unknown = encode_mapped_labels(
        ["426783006"], dictionary
    )
    assert not rhythm_complete
    assert not rhythm_unknown
    assert not rhythm_labels.any()

    labels, complete, acronyms, unknown = encode_mapped_labels(
        ["426783006", "164865005", "164909002"], dictionary
    )
    assert complete
    assert labels[SUPERCLASSES.index("MI")] == 1
    assert labels[SUPERCLASSES.index("CD")] == 1
    assert tuple(acronyms) == ("SR", "MI", "LBBB")
    assert not unknown

    _, incomplete, _, unmapped = encode_mapped_labels(["164865005", "999999"], dictionary)
    assert not incomplete
    assert unmapped == ("999999:UNREVIEWED",)
    assert EVALUATED_CLASSES == ("MI", "STTC", "CD", "HYP")


def test_units_and_antialiased_resampling_match_model_input():
    microvolts = np.ones((5000, 12), dtype=np.float32) * 1000
    millivolts = convert_to_mv(microvolts, ["uV"] * 12)
    assert np.allclose(millivolts, 1.0)
    downsampled = resample_500_to_100(millivolts)
    assert downsampled.shape == (1000, 12)
    assert np.allclose(downsampled[10:-10], 1.0, atol=1e-4)




def test_external_metric_summary_uses_the_named_superclass_columns():
    from scripts.evaluate_external_chapman import _metric_summary

    targets = np.zeros((8, len(SUPERCLASSES)), dtype=np.uint8)
    probabilities = np.full((8, len(SUPERCLASSES)), 0.1, dtype=np.float32)
    for row, name in enumerate(EVALUATED_CLASSES):
        index = SUPERCLASSES.index(name)
        targets[row, index] = 1
        probabilities[row, index] = 0.9
    metrics = _metric_summary(targets, probabilities, np.full(len(SUPERCLASSES), 0.5))
    assert metrics["macro_auroc"] == 1.0
    assert metrics["macro_f1"] == 1.0
    for name in EVALUATED_CLASSES:
        assert metrics["per_class"][name]["positive_records"] == 1
        assert metrics["per_class"][name]["auroc"] == 1.0



def test_official_sha256_manifest_verifier(tmp_path):
    import hashlib
    from scripts.download_chapman import verify_sha256_manifest

    payload = tmp_path / "RECORDS"
    payload.write_bytes(b"wfdb-record-list")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS.txt").write_text(f"{digest}  RECORDS\n", encoding="utf-8")
    assert verify_sha256_manifest(tmp_path, minimum_entries=1) == 1

