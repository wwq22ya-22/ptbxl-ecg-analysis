"""Check conservative LUDB-to-PTB-XL superclass decisions."""

from scripts.evaluate_external_ludb import encode_ludb_labels
from src.data.ptbxl import SUPERCLASSES


def _row(**changes):
    row = {
        "Conduction abnormalities": "",
        "Hypertrophies": "",
        "Ischemia": "",
        "Non-specific repolarization abnormalities": "",
        "Other states": "",
    }
    row.update(changes)
    return row


def _class(values, name):
    return bool(values[SUPERCLASSES.index(name)])


def test_explicit_conduction_and_hypertrophy_can_cooccur_without_implying_normal():
    labels, eligible, reasons = encode_ludb_labels(_row(**{
        "Conduction abnormalities": "Incomplete right bundle branch block",
        "Hypertrophies": "Left ventricular hypertrophy",
    }))
    assert _class(labels, "CD") and _class(labels, "HYP")
    assert _class(eligible, "CD") and _class(eligible, "HYP")
    assert not _class(eligible, "NORM")
    assert not reasons


def test_stemi_supports_mi_but_does_not_make_sttc_or_normal_labels():
    labels, eligible, _ = encode_ludb_labels(_row(**{"Ischemia": "Stemi: anterior wall"}))
    assert _class(labels, "MI") and _class(eligible, "MI")
    assert not _class(eligible, "STTC")
    assert not _class(eligible, "NORM")


def test_ambiguous_scar_is_unknown_unless_explicit_repolarization_is_present():
    labels, eligible, reasons = encode_ludb_labels(_row(**{
        "Ischemia": "Scar formation: septal",
        "Non-specific repolarization abnormalities": "Non-specific repolarization abnormalities: inferior wall",
    }))
    assert not _class(eligible, "MI")
    assert _class(labels, "STTC") and _class(eligible, "STTC")
    assert any("scar formation" in value for value in reasons)


def test_overload_alone_is_not_silently_called_hypertrophy_negative():
    labels, eligible, reasons = encode_ludb_labels(_row(**{
        "Hypertrophies": "Left ventricular overload",
    }))
    assert not _class(labels, "HYP") and not _class(eligible, "HYP")
    assert "HYP:left ventricular overload" in reasons


def test_unknown_conduction_label_is_excluded_from_cd_metric():
    _, eligible, reasons = encode_ludb_labels(_row(**{
        "Conduction abnormalities": "Sinoatrial blockade, undefined",
    }))
    assert not _class(eligible, "CD")
    assert reasons == ("CD:sinoatrial blockade, undefined",)
