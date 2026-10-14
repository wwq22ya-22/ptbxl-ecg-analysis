"""Meaningful checks for temporal alignment, shortcut use and training readiness."""
import io
import pytest
import torch
from src.models import build_model
from src.models.unet1d import ECGUNet1D, UpsampleFuse1D


@pytest.mark.parametrize('name,count',[('unet1d',579053),('unet_encoder1d',372053)])
def test_input_contract_logits_parameter_count_and_state_reload(name,count):
    torch.set_num_threads(4)
    model=build_model(name).eval()
    assert sum(p.numel() for p in model.parameters())==count
    x=torch.randn(2,12,1000)
    with torch.inference_mode():
        result=model(x)
        assert result.shape==(2,5) and torch.isfinite(result).all()
        single=torch.cat([model(row[None]) for row in x])
        torch.testing.assert_close(single,result,rtol=1e-4,atol=1e-5)
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        clone=build_model(name).eval();clone.load_state_dict(torch.load(buffer,weights_only=True))
        torch.testing.assert_close(clone(x),result,rtol=0,atol=0)


def test_four_decoder_stages_restore_non_power_of_two_lengths():
    model=ECGUNet1D().eval();x=torch.randn(2,12,1000)
    with torch.inference_mode():
        f=model.encoder(x)
        assert [v.shape[-1] for v in f]==[1000,500,250,125,62]
        decoded=f[-1]
        for layer,skip,n in zip(model.up,reversed(f[:-1]),[125,250,500,1000]):
            decoded=layer(decoded,skip);assert decoded.shape[-1]==n


def test_skip_branch_receives_gradient():
    block=UpsampleFuse1D(8,4)
    main=torch.randn(2,8,62,requires_grad=True);skip=torch.randn(2,4,125,requires_grad=True)
    block(main,skip).square().mean().backward()
    assert main.grad.abs().sum()>0 and skip.grad.abs().sum()>0


@pytest.mark.parametrize('name',['unet1d','unet_encoder1d'])
def test_weighted_multilabel_loss_updates_encoder_and_head(name):
    torch.manual_seed(42);model=build_model(name)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    before=model.encoder.stem[0].weight.detach().clone()
    labels=torch.tensor([[1.,0,1,0,0],[0,1,0,1,1]])
    loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.,3.,3.,3.,7.]))(model(torch.randn(2,12,1000)),labels)
    loss.backward()
    assert model.classifier[-1].weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step();assert not torch.equal(before,model.encoder.stem[0].weight)


@pytest.mark.parametrize('shape',[(2,1,1000),(2,12,999),(12,1000)])
def test_reject_wrong_waveform_contract(shape):
    with pytest.raises(ValueError):ECGUNet1D()(torch.zeros(shape))
