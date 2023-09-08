import torch
import torch.nn as nn


class LossForPretraining(nn.Module):
    # "The training loss is the sum of the mean masked LM likelihood
        # and the mean next sentence prediction likelihood."
    def __init__(self, vocab_size):
        super().__init__()

        self.vocab_size = vocab_size

        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_is_next, gt_is_next, pred_token_ids, gt_token_ids, select_mask):
        nsp_loss = self.ce(pred_is_next, gt_is_next)

        gt_token_ids[select_mask] = -100
        mlm_loss = self.ce(pred_token_ids.view(-1, self.vocab_size), gt_token_ids.view(-1))
        return nsp_loss, mlm_loss
