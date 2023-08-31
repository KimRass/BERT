# References
    # https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Literal


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int=5000) -> None:
        super().__init__()

        self.dim = dim

        pos = torch.arange(max_len).unsqueeze(1) # "$pos$"
        i = torch.arange(dim // 2).unsqueeze(0) # "$i$"
         # "$\sin(\text{pos} / 10000^{2 * i  / d_{\text{model}}})$"
        angle = pos / (10_000 ** (2 * i / dim))

        self.pe_mat = torch.zeros(size=(max_len, dim))
        self.pe_mat[:, 0:: 2] = torch.sin(angle) # "$text{PE}_(\text{pos}, 2i)$"
        self.pe_mat[:, 1:: 2] = torch.cos(angle) # "$text{PE}_(\text{pos}, 2i + 1)$"

        self.register_buffer("pos_enc_mat", self.pe_mat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, l, _ = x.shape
        x += self.pe_mat.unsqueeze(0)[:, : l, :].to(x.device)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, drop_prob=0.1):
        super().__init__()
    
        self.dim = dim # "$d_{model}$"
        self.n_heads = n_heads # "$h$"

        self.head_size = dim // n_heads # "$d_{k}$, $d_{v}$"

        self.q_proj = nn.Linear(dim, dim, bias=False) # "$W^{Q}_{i}$"
        self.k_proj = nn.Linear(dim, dim, bias=False) # "$W^{K}_{i}$"
        self.v_proj = nn.Linear(dim, dim, bias=False) # "$W^{V}_{i}$"

        self.attn_drop = nn.Dropout(drop_prob) # Not in the paper
        self.out_proj = nn.Linear(dim, dim, bias=False) # "$W^{O}$"

    def _get_attention_score(self, q, k):
        attn_score = torch.einsum("bnid,bnjd->bnij", q, k) # "MatMul" in "Figure 2" of the paper
        return attn_score

    def forward(self, q, k, v, mask=None):
        b, l, _ = q.shape

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q = q.view(b, self.n_heads, l, self.head_size)
        k = k.view(b, self.n_heads, l, self.head_size)
        v = v.view(b, self.n_heads, l, self.head_size)

        attn_score = self._get_attention_score(q=q, k=k)
        if mask is not None:
            attn_score.masked_fill_(mask=mask, value=-1e9) # "Mask (opt.)"
        attn_score /= (self.head_size ** 0.5) # "Scale"

        attn_weight = F.softmax(attn_score, dim=3) # "Softmax"
        attn_weight = self.attn_drop(attn_weight) # Not in the paper

        x = torch.einsum("bnij,bnjd->bnid", attn_weight, v) # "MatMul"
        x = rearrange(x, pattern="b n i d -> b i (n d)")

        x = self.out_proj(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, mlp_size, activ: Literal["relu", "gelu"]="relu", drop_prob=0.1):
        super().__init__()

        assert activ in ["relu", "gelu"],\
            """The argument `activ` must be one of (`"relu"`, `"gelu"`)"""

        self.dim = dim
        self.mlp_size = mlp_size
        self.activ = activ

        self.proj1 = nn.Linear(dim, self.mlp_size) # "$W_{1}$"
        if activ == "relu":
            self.relu = nn.ReLU()
        else:
            self.gelu = nn.GELU()
        self.proj2 = nn.Linear(self.mlp_size, dim) # "$W_{2}$"
        self.mlp_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj1(x)
        if self.activ == "relu":
            x = self.relu(x)
        else:
            x = self.gelu(x)
        x = self.proj2(x)
        x = self.mlp_drop(x) # Not in the paper
        return x


class ResidualConnection(nn.Module):
    def __init__(self, dim, drop_prob=0.1):
        super().__init__()

        self.dim = dim
        self.drop_prob = drop_prob

        self.norm = nn.LayerNorm(dim)
        self.resid_drop = nn.Dropout(drop_prob) # "Residual dropout"

    def forward(self, x, sublayer):
        skip = x.clone()
        x = self.norm(x)
        x = sublayer(x)
        x = self.resid_drop(x)
        x += skip
        return x


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size, pad_id=0):
        super().__init__(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=pad_id)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size, pad_id=0):
        super().__init__(num_embeddings=2, embedding_dim=embed_size, padding_idx=pad_id)


class PositionEmbedding(PositionalEncoding):
    def __init__(self, embed_size):
        super().__init__(dim=embed_size)


class TransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, mlp_size, attn_drop_prob=0.1, resid_drop_prob=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.mlp_size = mlp_size

        self.self_attn = MultiHeadAttention(dim=dim, n_heads=n_heads, drop_prob=attn_drop_prob)
        self.attn_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)
        self.feed_forward = PositionwiseFeedForward(dim=dim, mlp_size=mlp_size, activ="gelu")
        self.ff_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)

    def forward(self, x, mask=None):
        x = self.attn_resid_conn(x=x, sublayer=lambda x: self.self_attn(q=x, k=x, v=x, mask=mask))
        x = self.ff_resid_conn(x=x, sublayer=self.feed_forward)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size, mlp_size, attn_drop_prob, resid_drop_prob):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.mlp_size = mlp_size

        self.enc_stack = nn.ModuleList([
            TransformerLayer(
                n_heads=n_heads,
                dim=hidden_size,
                mlp_size=mlp_size,
                attn_drop_prob=attn_drop_prob,
                resid_drop_prob=resid_drop_prob,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, self_attn_mask):
        for enc_layer in self.enc_stack:
            x = enc_layer(x, mask=self_attn_mask)
        return x


def _get_pad_mask(seq, pad_id=0):
    mask = (seq == pad_id).unsqueeze(1).unsqueeze(3)
    return mask


class BERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers=12,
        n_heads=12,
        hidden_size=768,
        mlp_size=768 * 4,
        pad_id=0,
        embed_drop_prob=0.1,
        attn_drop_prob=0.1,
        resid_drop_prob=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.pad_id = pad_id
        self.embed_drop_prob = embed_drop_prob
        self.attn_drop_prob = attn_drop_prob
        self.resid_drop_prob = resid_drop_prob

        self.token_embed = TokenEmbedding(vocab_size=vocab_size, embed_size=hidden_size, pad_id=pad_id)
        self.seg_embed = SegmentEmbedding(embed_size=hidden_size, pad_id=pad_id)
        self.pos_embed = PositionEmbedding(embed_size=hidden_size)

        self.enmbed_drop = nn.Dropout(embed_drop_prob)

        self.tf_block = TransformerBlock(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            attn_drop_prob=attn_drop_prob,
            resid_drop_prob=resid_drop_prob,
        )

    def forward(self, seq, seg_ids):
        x = self.token_embed(seq)
        x = self.pos_embed(x)
        x += self.seg_embed(seg_ids)
        x = self.enmbed_drop(x)

        pad_mask = _get_pad_mask(seq=seq, pad_id=self.pad_id)
        x = self.tf_block(x, self_attn_mask=pad_mask)
        return x


# 110M parameters
class BERTBase(BERT):
    def __init__(self, vocab_size, pad_id=0):
        super().__init__(vocab_size=vocab_size, pad_id=pad_id)


# 340M parameters
class BERTLarge(BERT):
    def __init__(self, vocab_size, pad_id=0):
        super().__init__(
            vocab_size=vocab_size,
            n_layers=24,
            n_heads=16,
            hidden_size=1024,
            pad_id=pad_id
        )


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size=768, n_classes=1000):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_classed = n_classes

        self.cls_proj = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x[:, 0, :]
        x = self.cls_proj(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, drop_prob=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.cls_proj = nn.Linear(hidden_size, vocab_size)
        self.head_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.cls_proj(x)
        x = self.head_drop(x)
        return x


class NSPHead(nn.Module):
    def __init__(self, hidden_size=768, drop_prob=0.1):
        super().__init__()

        self.hidden_size = hidden_size

        self.cls_proj = nn.Linear(hidden_size, 2)
        self.head_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = x[:, 0, :]
        x = self.cls_proj(x)
        x = self.head_drop(x)
        return x


class BERTBaseForPretraining(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """
    def __init__(self, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()

        self.bert = BERTBase(vocab_size=vocab_size)

        self.nsp_head = NSPHead(self.bert.hidden_size)
        self.mlm_head = MLMHead(
            vocab_size=self.bert.vocab_size, hidden_size=self.bert.hidden_size,
        )

    def forward(self, seq, seg_ids):
        x = self.bert(seq=seq, seg_ids=seg_ids)
        return self.nsp_head(x), self.mlm_head(x)


class QuestionAnsweringHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # "We only introduce a start vector $S \in \mathbb{R}^{H}$ and an end vector
        # $E \in \mathbb{R}^{H}$ during fine-tuning."
        self.proj = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # "The probability of word $i$ being the start of the answer span is computed
        # as a dot product between $T_{i}$ and $S$ followed by a softmax over all of the words in the paragraph."
        x = self.proj(x)
        start_logit, end_logit = torch.split(x, split_size_or_sections=1, dim=2)
        start_logit, end_logit = start_logit.squeeze(), end_logit.squeeze()
        start_id, end_id = torch.argmax(start_logit, dim=1), torch.argmax(end_logit, dim=1)
        return start_id, end_id


class MultipleChoiceHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.proj(x)
        x = x.squeeze()
        x = torch.argmax(x, dim=1)
        return x


if __name__ == "__main__":
    HIDDEN_DIM = 768
    VOCAB_SIZE = 30_522

    BATCH_SIZE = 8
    SEQ_LEN = 512
    seq = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    sent1_len = torch.randint(low=2, high=SEQ_LEN + 1, size=(BATCH_SIZE,))
    seg_ids = torch.as_tensor([[0] * i + [1] * (SEQ_LEN - i) for i in sent1_len], dtype=torch.int64)

    model = BERTBase(vocab_size=VOCAB_SIZE)
    # model = BERTLarge(vocab_size=VOCAB_SIZE)
    output = model(seq=seq, seg_ids=seg_ids)
    print(output.shape)
