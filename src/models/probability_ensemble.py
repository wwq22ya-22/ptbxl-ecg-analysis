"""Frozen probability-average teachers for multi-label distillation."""
from pathlib import Path
import torch
from torch import nn


class ProbabilityEnsemble(nn.Module):
    """Return logits of the equally averaged independent-label probabilities."""
    def __init__(self, members):
        super().__init__()
        if not members:
            raise ValueError("A teacher ensemble needs at least one member.")
        self.members = nn.ModuleList(members)
        self.requires_grad_(False)
        self.eval()

    def forward(self, signals):
        probability = torch.stack([member(signals).sigmoid() for member in self.members]).mean(0)
        return torch.logit(probability, eps=1e-7)


def load_teacher(path: str | Path, device="cpu"):
    from src.models import build_model
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if cp["model_type"] == "probability_ensemble":
        members = []
        for member in cp["members"]:
            model = build_model(member["model_type"], dropout=member["dropout"])
            model.load_state_dict(member["model_state"])
            members.append(model)
        model = ProbabilityEnsemble(members)
    else:
        model = build_model(cp["model_type"], dropout=cp["config"]["model"]["dropout"])
        model.load_state_dict(cp["model_state"])
    return model.to(device).eval().requires_grad_(False), cp
