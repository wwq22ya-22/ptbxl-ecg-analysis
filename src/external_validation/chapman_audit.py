"""Versioned SNOMED audit; absence is never a verified negative diagnosis.

Primary labels contain explicit positives only. A second, clearly conditional
mask allows absence-as-negative on morphology-bearing records, excluding
uncertain concepts class by class. It is for sensitivity analysis, not proof
that the source has exhaustive annotations.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data.ptbxl import SUPERCLASSES

MAPPING_VERSION = "chapman-snomed-positive-v1-20261001"
FOUR_CLASSES = ("MI", "STTC", "CD", "HYP")

# Exact concepts in the official Challenge 2021 dictionaries and the native
# Chapman dictionary. Do not infer a target from an ambiguous acronym.
POSITIVE_CODES = {
    "MI": "164865005 57054005 54329005 164867002".split(),
    "STTC": "429622005 164931005 164934002 59931005 111975006 428750005 55930002".split(),
    "CD": ("270492004 195042002 54016002 28189009 27885002 233917008 "
           "164909002 59118001 733534002 713427006 713426002 251120003 "
           "445118002 445211001 698252002 6374002 195060002 74390002 426183003").split(),
    "HYP": "164873001 89792004 446358003 446813000 67741000119109 266249003 195126007".split(),
}
CODE_POSITIVES = {code: name for name, codes in POSITIVE_CODES.items() for code in codes}

RHYTHM_CODES = frozenset((
    "251173003 284470004 164889003 164890007 713422000 233896004 251166008 "
    "233897008 426995002 251164006 427393009 426177001 426783006 427084000 "
    "426761007 11157007 75532003 13640000 17338001 251180001 195101003 "
    "17366009 233892002 251187003 61277005 426664006 106068003 29320008 "
    "251170000 426627000 63593006 427172004 81898007 164896001 111288001 "
    "425856008 426648003 164895002 65778007 5609005"
).split())

# Uncertain evidence affects the indicated target(s), not every class.
UNCERTAIN_CODES = {
    "164917005": ("MI",),             # abnormal Q wave is not necessarily MI
    "365413008": ("MI",),             # poor R progression is not diagnostic MI
    "428417006": ("STTC",),           # early repolarization: superclass ambiguity
    "164930006": ("STTC",),           # ST interval abnormal, not explicit ST/T change
    "77867006": ("STTC",),            # short QT is not PTB-XL LNGQT
    "164937009": ("STTC",),           # U-wave abnormality
    "418818005": ("STTC", "CD"),    # Brugada lacks equivalent target definition
    "164947007": ("CD",),             # prolonged PR without explicit AV block
    "50799005": ("CD",),              # AV dissociation does not establish AV block
    "49578007": ("CD",),              # short PR alone is not WPW
    "164942001": ("CD",),             # fragmented QRS
    "55827005": ("HYP",),             # high voltage alone is not hypertrophy
    "67751000119106": ("HYP",),       # atrial high voltage
    "164912004": ("HYP",),            # P-wave change without hypertrophy diagnosis
    "251205003": ("HYP",),            # prolonged P wave
    "251223006": ("HYP",),            # tall P wave
    "10370003": FOUR_CLASSES,          # paced waveform may obscure morphology
}
# Reviewed extra morphology, not a positive in the four target superclasses.
NEUTRAL_CODES = frozenset("39732003 47665007 251199005 251198002 61721007 251146004 251148003 251147008".split())


def load_catalog(paths: list[Path]) -> dict[str, dict]:
    """Load names from official CSV files; preserve source and conflicting aliases."""
    catalog: dict[str, dict] = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                code = str(row.get("SNOMEDCTCode", row.get("Snomed_CT", ""))).strip()
                name = row.get("Dx", row.get("Full Name", ""))
                acronym = row.get("Abbreviation", row.get("Acronym Name", ""))
                if not code:
                    continue
                item = catalog.setdefault(code, {"names": [], "acronyms": [], "sources": []})
                for key, value in (("names", name), ("acronyms", acronym), ("sources", path.name)):
                    if value not in item[key]:
                        item[key].append(value)
    return catalog


def rule_for(code: str) -> tuple[str, tuple[str, ...]]:
    if code in CODE_POSITIVES:
        return "explicit_positive", (CODE_POSITIVES[code],)
    if code in RHYTHM_CODES:
        return "rhythm_only", ()
    if code in UNCERTAIN_CODES:
        return "uncertain", UNCERTAIN_CODES[code]
    if code in NEUTRAL_CODES:
        return "reviewed_other_morphology", ()
    return "unreviewed", FOUR_CLASSES


def encode_audited_labels(codes: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return targets, positive-only mask, conditional mask, and audit reasons.

    A positive remains usable when other codes are unknown. Conditional
    negatives require non-rhythm evidence and no uncertainty for that class.
    NORM never has an eligible label. Rhythm-only records stay all unknown.
    """
    labels = np.zeros(len(SUPERCLASSES), dtype=np.uint8)
    uncertain = set()
    morphology = False
    reasons = []
    for code in codes:
        role, classes = rule_for(code)
        if role == "explicit_positive":
            labels[SUPERCLASSES.index(classes[0])] = 1
            morphology = True
        elif role != "rhythm_only":
            morphology = True
            uncertain.update(classes)
            if role in {"uncertain", "unreviewed"}:
                reasons.append(f"{role}:{code}:{','.join(classes)}")
    primary = labels.astype(bool)
    conditional = primary.copy()
    if morphology:
        for name in FOUR_CLASSES:
            index = SUPERCLASSES.index(name)
            conditional[index] |= name not in uncertain
    return labels, primary, conditional, tuple(reasons)
