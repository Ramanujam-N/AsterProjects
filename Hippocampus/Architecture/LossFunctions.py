import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-7

    def forward(self, x, target):
        smooth = 1
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x.double() * target, dims)
        cardinality = torch.sum(x.double() + target, dims)

        dice_score = ((2. * intersection + smooth) / (cardinality + smooth))
        
        return (1 - dice_score).mean()
