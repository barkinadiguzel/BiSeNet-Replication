import torch.nn as nn

class BiSeNetLoss(nn.Module):
    def __init__(self, aux_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.aux_weight = aux_weight

    def forward(self, pred, target, aux1=None, aux2=None):
        loss = self.ce(pred, target)

        if aux1 is not None:
            loss += self.aux_weight * self.ce(aux1, target)
        if aux2 is not None:
            loss += self.aux_weight * self.ce(aux2, target)

        return loss
