import torch.nn as nn
import torch.nn.functional as F


class LossForPretraining(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_is_next, gt_is_next, pred_token_ids, gt_token_ids,):
        mlm_loss = F.cross_entropy(
            pred_token_ids.permute(0, 2, 1), gt_token_ids, reduction="mean",
        )

        nsp_loss = F.cross_entropy(pred_is_next, gt_is_next)
        return nsp_loss, mlm_loss
