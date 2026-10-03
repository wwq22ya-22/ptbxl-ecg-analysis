import numpy as np

from src.data.ptbxl import SUPERCLASSES
from src.external_validation.chapman_audit import encode_audited_labels


def test_sinus_rhythm_is_not_normal_and_has_no_verified_negatives():
    y, strict, conditional, _ = encode_audited_labels(('426783006',))
    assert not y.any() and not strict.any() and not conditional.any()


def test_explicit_positive_survives_unreviewed_other_code():
    y, strict, conditional, reasons = encode_audited_labels(('164873001', '999999'))
    expected = np.array([False, False, False, False, True])
    assert np.array_equal(strict, expected)
    assert np.array_equal(conditional, expected)
    assert y[SUPERCLASSES.index('HYP')] == 1
    assert reasons == ('unreviewed:999999:MI,STTC,CD,HYP',)


def test_ambiguous_q_wave_masks_mi_but_keeps_explicit_cd():
    y, strict, conditional, _ = encode_audited_labels(('164917005', '59118001'))
    assert not conditional[SUPERCLASSES.index('MI')]
    assert strict[SUPERCLASSES.index('CD')]
    assert conditional[SUPERCLASSES.index('STTC')]
    assert not strict[SUPERCLASSES.index('STTC')]
    assert not conditional[SUPERCLASSES.index('NORM')]


def test_high_voltage_does_not_invent_hypertrophy_or_normal():
    y, strict, conditional, _ = encode_audited_labels(('55827005',))
    assert not y.any() and not strict.any()
    assert not conditional[SUPERCLASSES.index('HYP')]
    assert not conditional[SUPERCLASSES.index('NORM')]


def test_positive_only_metrics_do_not_report_auc_or_precision():
    from scripts.evaluate_audited_chapman import summarize_metrics
    targets = np.array([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.uint8)
    strict = targets.astype(bool)
    scores = np.array([[.1, .9, .1, .1, .1], [.1, .2, .1, .1, .1]])
    result = summarize_metrics(targets, strict, scores, np.full(5, .5), mode='positive_only')
    mi = result['per_class']['MI']
    assert mi['recall'] == .5 and mi['true_positive'] == 1 and mi['false_negative'] == 1
    assert mi['auroc'] is None and mi['precision'] is None and mi['f1'] is None
    assert result['macro_f1'] is None
