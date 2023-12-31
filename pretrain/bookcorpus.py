# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html

import os
import torch
from torch.utils.data import Dataset
import random

import config
from pretrain.wordpiece import load_fast_bert_tokenizer
from utils import _token_ids_to_segment_ids
from pretrain.wordpiece import parse

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _encode(x, tokenizer):
    encoding = tokenizer(
        x,
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    if isinstance(x, str):
        return encoding["input_ids"][1: -1]
    else:
        return [token_ids[1: -1] for token_ids in encoding["input_ids"]]


# "For the pre-training corpus we use the BookCorpus (800M words) (Zhu et al., 2015)
# and English Wikipedia (2,500M words)."
# "For Wikipedia we extract only the text passages and ignore lists, tables, and headers.
# It is critical to use a document-level corpus rather than a shuffled sentence-level corpus
# such as the Billion Word Benchmark."
class BookCorpusForBERT(Dataset):
    def __init__(
        self,
        epubtxt_dir,
        tokenizer,
        seq_len,
    ):
        self.epubtxt_dir = epubtxt_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.unk_id = tokenizer.unk_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        self.lines = parse(epubtxt_dir)

    def _sample_latter_sentence(self, idx):
        if random.random() < 0.5:
            latter_idx = idx + 1
            is_next = 1
        else:
            latter_idx = random.randrange(len(self.lines))
            is_next = 0
        latter_line = self.lines[latter_idx]
        return latter_line, torch.as_tensor(is_next)

    def _to_bert_input(self, former_token_ids, latter_token_ids):
        token_ids = former_token_ids[: self.seq_len - 2]
        # Add "[CLS]" and the first "[SEP]" tokens.
        token_ids = [self.cls_id] + token_ids + [self.sep_id]
        if len(token_ids) >= self.seq_len:
            token_ids = token_ids[: self.seq_len]
        else:
            if len(token_ids) < self.seq_len - 1:
                token_ids += latter_token_ids
                token_ids = token_ids[: self.seq_len - 1]
                token_ids += [self.sep_id] # Add the second "[SEP]" token.
            token_ids += [self.pad_id] * (self.seq_len - len(token_ids)) # Pad.
        return torch.as_tensor(token_ids)

    def __len__(self):
        return len(self.lines) - 1

    def __getitem__(self, idx):
        former_line = self.lines[idx]
        former_token_ids = _encode(former_line, tokenizer=self.tokenizer)
        latter_line, is_next = self._sample_latter_sentence(idx)
        latter_token_ids = _encode(latter_line, tokenizer=self.tokenizer)

        token_ids = self._to_bert_input(
            former_token_ids=former_token_ids, latter_token_ids=latter_token_ids,
        )
        seg_ids = _token_ids_to_segment_ids(token_ids=token_ids, sep_id=self.sep_id)
        return token_ids, seg_ids, is_next
