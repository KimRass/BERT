import torch

    
def get_nsp_acc(pred_is_next, gt_is_next):
    argmax = torch.argmax(pred_is_next, dim=1)
    # acc = (gt_is_next == argmax).sum() / gt_is_next.numel()
    acc = (gt_is_next == argmax).float().mean()
    return acc.item()


def get_mlm_acc(pred_token_ids, gt_token_ids):
    argmax = torch.argmax(pred_token_ids, dim=2)
    acc = (gt_token_ids == argmax).sum() / gt_token_ids.numel()
    return acc.item()
