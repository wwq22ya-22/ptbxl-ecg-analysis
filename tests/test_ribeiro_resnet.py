"""Check the author's shortcut semantics and adapted SAME downsampling."""
import torch
from src.models import build_model
from src.models.ribeiro_resnet import RibeiroResidualUnit, SameMaxPool1D


def test_same_pool_alignment_including_partial_windows():
    x = torch.arange(9, dtype=torch.float32).reshape(1, 1, 9)
    torch.testing.assert_close(SameMaxPool1D(4)(x), torch.tensor([[[2., 6., 8.]]]))


def test_shortcut_carries_raw_sum_before_bn_relu_dropout():
    block = RibeiroResidualUnit(2, 2, stride=1, dropout=0).eval()
    with torch.no_grad():
        block.main_first[0].conv.weight.zero_()
        block.main_second.conv.weight.zero_()
    x = torch.randn(2, 2, 13)
    y = -torch.ones_like(x)
    processed, shortcut = block((x, y))
    torch.testing.assert_close(shortcut, y, rtol=0, atol=0)
    torch.testing.assert_close(processed, torch.zeros_like(y), rtol=0, atol=0)


def test_adapted_time_dimensions_and_batch_independent_eval():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = build_model('ribeiro_resnet100').eval()
    signals = torch.randn(2, 12, 1000)
    with torch.inference_mode():
        x = model.stem(signals); streams = (x, x)
        for block, length in zip(model.blocks, [250, 63, 32, 16]):
            streams = block(streams)
            assert streams[0].shape[-1] == streams[1].shape[-1] == length
        batch = model(signals)
        single = torch.cat([model(row[None]) for row in signals])
        torch.testing.assert_close(single, batch, rtol=1e-4, atol=1e-5)
