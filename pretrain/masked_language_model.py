# References
    # https://nn.labml.ai/transformers/mlm/index.html

import torch

import config
from pretrain.wordpiece import load_bert_tokenizer


class MaskedLanguageModel(object):
    def __init__(
        self,
        vocab_size,
        mask_id,
        no_mask_token_ids=[],
        select_prob=0.15,
        mask_prob=0.8,
        randomize_prob=0.1,
    ):
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.no_mask_token_ids = no_mask_token_ids
        self.select_prob = select_prob
        self.mask_prob = mask_prob
        self.randomize_prob = randomize_prob

        if mask_id not in no_mask_token_ids:
            no_mask_token_ids += [mask_id]


    def __call__(self, gt_token_ids):
        masked_token_ids = gt_token_ids.clone()

        rand_tensor = torch.rand(masked_token_ids.shape, device=masked_token_ids.device)
        no_mask_mask = torch.isin(
            masked_token_ids,
            torch.as_tensor(self.no_mask_token_ids, device=masked_token_ids.device),
        )
        rand_tensor.masked_fill_(mask=no_mask_mask, value=1)

        # "Chooses 15% of the token positions at random for prediction."
        select_mask = (rand_tensor < self.select_prob)
        # `select_mask.sum() / gt_token_ids.numel() ~= 0.15`

        # "If the $i$-th token is chosen, we replace the $i$-th token with (1) the [MASK] token
        # 80% of the time."
        rand_tensor = torch.rand(masked_token_ids.shape, device=masked_token_ids.device)
        mask_mask = select_mask & (rand_tensor < self.mask_prob)
        # `mask_mask.sum() / select_mask.sum() ~= 0.8`
        masked_token_ids.masked_fill_(mask=mask_mask, value=self.mask_id)

        # "(2) a random token 10% of the time
        # (3) the unchanged $i$-th token 10% of the time."
        rand_tensor = torch.rand(masked_token_ids.shape, device=masked_token_ids.device)
        randomize_mask = select_mask &\
            (rand_tensor >= self.mask_prob) &\
            (rand_tensor < (self.mask_prob + self.randomize_prob))
        # `randomize_mask.sum() / select_mask.sum() ~= 0.1`
        random_token_ids = torch.randint(
            high=self.vocab_size,
            size=torch.Size((randomize_mask.sum(),)),
            device=masked_token_ids.device,
        )
        masked_token_ids[randomize_mask.nonzero(as_tuple=True)] = random_token_ids
        return masked_token_ids, select_mask


if __name__ == "__main__":
    tokenizer = load_bert_tokenizer(config.VOCAB_PATH)
    mlm = MaskedLanguageModel(
        vocab_size=config.VOCAB_SIZE,
        mask_id=tokenizer.token_to_id("[MASK]"),
        select_prob=config.SELECT_PROB,
        mask_prob=config.MASK_PROB,
        randomize_prob=config.RANDOMIZE_PROB,
    )