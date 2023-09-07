import torch
import torch.nn as nn


class LossForPretraining(nn.Module):
    # "The training loss is the sum of the mean masked LM likelihood
        # and the mean next sentence prediction likelihood."
    def __init__(self, vocab_size, smoothing=0):
        super().__init__()

        assert 0 <= smoothing <= 1, "The argument `smoothing` must be between 0 and 1!"

        self.vocab_size = vocab_size
        self.smoothing = smoothing

        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, pred_is_next, gt_is_next, pred_token_ids, gt_token_ids):
        nsp_loss = self.ce(pred_is_next, gt_is_next)

        new_gt_token_ids = torch.full_like(
            pred_token_ids, fill_value=self.smoothing / (self.vocab_size - 1),
        )
        new_gt_token_ids.scatter_(1, gt_token_ids.unsqueeze(1), 1 - self.smoothing)
        mlm_loss = self.ce(pred_token_ids.permute(0, 2, 1), new_gt_token_ids)
        return nsp_loss, mlm_loss
