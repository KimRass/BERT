import torch.nn as nn
import torch.nn.functional as F


class PretrainingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mlm_pred, nsp_pred, token_ids, is_next):
        raw_mlm_loss = F.cross_entropy(
            mlm_pred.permute(0, 2, 1), token_ids, reduction="none",
        )
        # raw_mlm_loss *= gt["prediction_target"]
        mlm_loss = raw_mlm_loss.sum()

        nsp_loss = F.cross_entropy(nsp_pred, is_next)
        # loss = mlm_loss + nsp_loss
        # return loss
        return nsp_loss, mlm_loss
