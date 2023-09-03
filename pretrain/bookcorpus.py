# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html

# "For the pre-training corpus we use the BookCorpus (800M words) (Zhu et al., 2015)
# and English Wikipedia (2,500M words)."
# "For Wikipedia we extract only the text passages and ignore lists, tables, and headers.
# It is critical to use a document-level corpus rather than a shuffled sentence-level corpus
# such as the Billion Word Benchmark."
import os
import torch
from torch.utils.data import Dataset
import random
from tqdm.auto import tqdm

from pretrain.wordpiece import parse

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class BookCorpusDataset(Dataset):
    def __init__(
        self,
        epubtxt_dir,
        tokenizer,
        seq_len,
        chunk_size=2 ** 14,
    ):
        self.epubtxt_dir = epubtxt_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.chunk_size = chunk_size

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")

        self.lines = parse(epubtxt_dir)
        self._tokenize()

    def _tokenize(self):
        print("Tokenizing BookCorpus...")
        self.ls_token_ids = list()
        for idx in tqdm(range(0, len(self.lines), self.chunk_size)):
            encoded = self.tokenizer.encode_batch(self.lines[idx: idx + self.chunk_size])
            self.ls_token_ids.extend([i.ids for i in encoded])
        print("Completed")

    def _to_bert_input(self, former_token_ids, latter_token_ids):
        ### Add '[CLS]' and '[SEP]' tokens.
        token_ids = (
            [self.cls_id] + former_token_ids[: self.seq_len - 3] + [self.sep_id] + latter_token_ids
        )[: self.seq_len - 1] + [self.sep_id]
        ### Pad.
        token_ids += [self.pad_id] * (self.seq_len - len(token_ids))
        return torch.as_tensor(token_ids)

    def _sample_latter_sentence(self, idx):
        if random.random() < 0.5:
            latter_idx = idx + 1
            is_next = 1
        else:
            latter_idx = random.randrange(len(self.lines))
            is_next = 0
        latter_token_ids = self.ls_token_ids[latter_idx]
        return latter_token_ids, torch.as_tensor(is_next)

    def _token_ids_to_segment_ids(self, token_ids):
        seg_ids = torch.zeros_like(token_ids, dtype=token_ids.dtype, device=token_ids.device)
        is_sep = (token_ids == self.sep_id)
        if is_sep.sum() == 2:
            a, b = is_sep.nonzero()
            seg_ids[a + 1: b + 1] = 1
        return seg_ids

    def __len__(self):
        return len(self.ls_token_ids) - 1

    def __getitem__(self, idx):
        former_token_ids = self.ls_token_ids[idx]
        latter_token_ids, is_next = self._sample_latter_sentence(idx)

        token_ids = self._to_bert_input(
            former_token_ids=former_token_ids, latter_token_ids=latter_token_ids,
        )
        seg_ids = self._token_ids_to_segment_ids(token_ids)
        return token_ids, seg_ids, is_next
