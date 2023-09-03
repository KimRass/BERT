import torch.nn as nn


class LossForPretraining(nn.Module):
    # "The training loss is the sum of the mean masked LM likelihood
        # and the mean next sentence prediction likelihood."
    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, pred_is_next, gt_is_next, pred_token_ids, gt_token_ids):
        nsp_loss = self.ce(pred_is_next, gt_is_next)
        mlm_loss = self.ce(pred_token_ids.permute(0, 2, 1), gt_token_ids)
        return nsp_loss, mlm_loss
