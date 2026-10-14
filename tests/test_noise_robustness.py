import numpy as np
import pytest

from src.training.noise import add_raw_gaussian_noise, snr_db, NoiseViewsDataset


def test_raw_noise_is_reproducible_per_lead_and_preserves_input():
    source = np.random.default_rng(4).normal(size=(1000,12)).astype(np.float32)
    source[:,0] *= 100
    snapshot = source.copy()
    corrupted, noise = add_raw_gaussian_noise(source, 20, seed=17)
    np.testing.assert_array_equal(source, snapshot)
    np.testing.assert_array_equal(corrupted, add_raw_gaussian_noise(source,20,seed=17)[0])
    np.testing.assert_allclose(snr_db(source, noise),20,atol=1e-3)
    assert not np.array_equal(corrupted, add_raw_gaussian_noise(source,20,seed=18)[0])


def test_flat_lead_stays_finite_without_inventing_signal_power():
    source = np.zeros((1000,12),dtype=np.float32)
    corrupted,noise = add_raw_gaussian_noise(source,10,seed=5)
    np.testing.assert_array_equal(corrupted,source)
    assert np.isfinite(corrupted).all()
    assert np.isnan(snr_db(source,noise)).all()


def test_distillation_selects_same_record_and_view_for_teacher_and_student():
    views = [np.full((4,12,1000),v,dtype=np.float32) for v in range(3)]
    targets=np.eye(5,dtype=np.float32)[:4]
    logits=np.stack([np.full((4,5),v,dtype=np.float32) for v in range(3)])
    data=NoiseViewsDataset(views,targets,[1,2,3,4],probabilities=[.5,.25,.25],seed=44,
                           teacher_logits=logits,teacher_ids=[1,2,3,4])
    for epoch in [1,2,3]:
        data.set_epoch(epoch)
        for index in range(4):
            x,y,teacher=data[index]
            assert x[0,0].item()==teacher[0].item()==data.selected_views[index]
            np.testing.assert_array_equal(y.numpy(),targets[index])
    for index,view in enumerate(views):
        assert np.all(view==index)
    with pytest.raises(ValueError,match='exact order'):
        NoiseViewsDataset(views,targets,[1,2,3,4],probabilities=[.5,.25,.25],seed=44,
                           teacher_logits=logits,teacher_ids=[4,3,2,1])


def test_clean_control_never_selects_augmented_data_and_epochs_are_deterministic():
    views=[np.zeros((100,12,1000),dtype=np.float32)]*3
    control=NoiseViewsDataset(views,np.zeros((100,5)),np.arange(100),probabilities=[1,0,0],seed=44)
    control.set_epoch(10)
    assert np.all(control.selected_views==0)
    noise=NoiseViewsDataset(views,np.zeros((100,5)),np.arange(100),probabilities=[.5,.25,.25],seed=44)
    noise.set_epoch(1)
    before=noise.selected_views.copy()
    noise.set_epoch(1)
    np.testing.assert_array_equal(noise.selected_views,before)
    noise.set_epoch(2)
    assert not np.array_equal(noise.selected_views,before)
