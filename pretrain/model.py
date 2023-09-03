import torch.nn as nn

from model import BERT


class MLMHead(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, drop_prob=0.1):
        super().__init__()

        self.proj = nn.Linear(hidden_size, vocab_size)
        self.head_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj(x)
        # x = self.head_drop(x)
        return x


class NSPHead(nn.Module):
    def __init__(self, hidden_size=768, drop_prob=0.1):
        super().__init__()

        self.proj = nn.Linear(hidden_size, 2)
        self.head_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        # print(x.shape)
        x = x[:, 0, :]
        # print(x.shape)
        x = self.proj(x)
        # x = self.head_drop(x)
        return x


class BERTForPretraining(nn.Module):
    def __init__(self, vocab_size, max_len, pad_id, n_layers, n_heads, hidden_size, mlp_size):
        super().__init__()

        self.bert = BERT(
            vocab_size=vocab_size,
            max_len=max_len,
            pad_id=pad_id,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
        )

        self.nsp_head = NSPHead(hidden_size=self.bert.hidden_size)
        # self.mlm_head = MLMHead(
        #     vocab_size=self.bert.vocab_size, hidden_size=self.bert.hidden_size,
        # )

    def forward(self, token_ids, seg_ids):
        # print(token_ids)
        # print(seg_ids)
        x = self.bert(token_ids=token_ids, seg_ids=seg_ids)
        pred_is_next = self.nsp_head(x)
        # pred_token_ids = self.mlm_head(x)
        # return pred_is_next, pred_token_ids
        return pred_is_next
