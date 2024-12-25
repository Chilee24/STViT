import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy as SoftTargetCrossEntropy_timm
from timm.loss import LabelSmoothingCrossEntropy as LabelSmoothingCrossEntropy_timm

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert smoothing < 1.
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        
        logprobs = F.log_softmax(x, dim=-1)
        if len(target.shape) == 2: # one-hot encoding
            target = target.argmax(-1)
        
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(-1))
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        loss = torch.sum(-target * logprobs, dim=-1)
        return loss.mean()