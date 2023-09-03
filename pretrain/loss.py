import torch
import torch.nn as nn
import torch.nn.functional as F


class LossForPretraining(nn.Module):
    # "The training loss is the sum of the mean masked LM likelihood
        # and the mean next sentence prediction likelihood."
    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(reduction="mean")

    # def forward(self, pred_is_next, gt_is_next, pred_token_ids, gt_token_ids):
    def forward(self, pred_is_next, gt_is_next):
        # argmax = torch.argmax(pred_is_next, dim=-1)
        # print(argmax)
        # print(gt_is_next, end="\n\n")
        nsp_loss = self.ce(pred_is_next, gt_is_next)
        # mlm_loss = self.ce(pred_token_ids.permute(0, 2, 1), gt_token_ids)
        # return nsp_loss, mlm_loss
        return nsp_loss

if __name__ == "__main__":
    import torch
    pred = torch.Tensor([[1.2275, 0.8730]])
    # pred = F.softmax(pred, dim=-1)
    gt = torch.Tensor([1]).long()
    # pred.shape, gt.shape
    nn.CrossEntropyLoss(reduction="mean")(pred, gt)
    
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    input = torch.randn(3, requires_grad=True)
    m(input)
    target = torch.empty(3).random_(2)
    output = loss(m(input), target)
    output
    target
    # output.backward()