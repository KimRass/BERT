# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html

# "For the pre-training corpus we use the BookCorpus (800M words) (Zhu et al., 2015)
# and English Wikipedia (2,500M words)."
# "For Wikipedia we extract only the text passages and ignore lists, tables, and headers.
# It is critical to use a document-level corpus rather than a shuffled sentence-level corpus
# such as the Billion Word Benchmark."

import torch
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
from tqdm.auto import tqdm
import csv

import config
from pretrain.wordpiece import train_bert_tokenizer, load_bert_tokenizer


def _parse_and_tokenize(epubtxt_dir, tokenizer):
    ls_token_ids = list()
    for doc_path in tqdm(list(Path(epubtxt_dir).glob("*.txt"))):
        parags = list()
        for parag in open(doc_path, mode="r", encoding="utf-8"):
            parag = parag.strip()
            if parag == "":
                continue

            parags.append(parag)
        encoded = tokenizer.encode_batch(parags)
        ls_token_ids.extend([i.ids[1: -1] for i in encoded])
    return ls_token_ids


def save_token_ids(epubtxt_dir, tokenizer, csv_path):
    print("Tokenizing BookCorpus paragraphs...")
    ls_token_ids = _parse_and_tokenize(epubtxt_dir=epubtxt_dir, tokenizer=tokenizer)
    if not Path(csv_path).exists():
        with open(csv_path, mode="w") as f:
            writer = csv.writer(f)
            writer.writerows(ls_token_ids)


def load_token_ids(csv_path):
    ls_token_ids = list()
    with open(csv_path, mode="r") as f:
        reader = csv.reader(f)
        for row in reader:
            ls_token_ids.append(list(map(int, row)))
    return ls_token_ids


class BookCorpusForBERT(Dataset):
    def __init__(
        self,
        epubtxt_dir,
        csv_path,
        tokenizer,
        max_len,
    ):
        self.epubtxt_dir = epubtxt_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")

        if not Path(csv_path).exists():
            save_token_ids(epubtxt_dir=epubtxt_dir, tokenizer=tokenizer, csv_path=csv_path)
        self.ls_token_ids = load_token_ids(csv_path)

    def _to_bert_input(self, prev_token_ids, next_token_ids):
        token_ids = (
            [self.cls_id] + prev_token_ids[: self.max_len - 3] + [self.sep_id] + next_token_ids
        )[: self.max_len - 1] + [self.sep_id]
        token_ids += [self.pad_id] * (self.max_len - len(token_ids))
        return token_ids

    def _sample_next_sentence(self, idx):
        if random.random() < 0.5:
            next_idx = idx + 1
            is_next = 1
        else:
            next_idx = random.randrange(len(self.ls_token_ids))
            is_next = 0
        next_token_ids = self.ls_token_ids[next_idx]
        return next_token_ids, torch.as_tensor(is_next)

    def _token_ids_to_segment_ids(self, token_ids):
        seg_ids = torch.zeros_like(token_ids, dtype=token_ids.dtype, device=token_ids.device)
        is_sep = (token_ids == self.sep_id)
        if is_sep.sum() == 2:
            a, b = is_sep.nonzero()
            seg_ids[a + 1: b + 1] = 1
        return seg_ids

    def __len__(self):
        return len(self.ls_token_ids)

    def __getitem__(self, idx):
        prev_token_ids = self.ls_token_ids[idx]
        next_token_ids, is_next = self._sample_next_sentence(idx)
        token_ids = self._to_bert_input(
            prev_token_ids=prev_token_ids, next_token_ids=next_token_ids,
        )
        token_ids = torch.as_tensor(token_ids)
        seg_ids = self._token_ids_to_segment_ids(token_ids)
        return token_ids, seg_ids, is_next


if __name__ == "__main__":
    epubtxt_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus/epubtxt"
    csv_path = "/Users/jongbeomkim/Desktop/workspace/bert_from_scratch/pretrain/bookcorpus_token_ids.csv"
    ds = BookCorpusForBERT(
        # epubtxt_dir=config.EPUBTXT_DIR, tokenizer=tokenizer, max_len=config.MAX_LEN
        epubtxt_dir=epubtxt_dir, csv_path=csv_path, tokenizer=tokenizer, max_len=config.MAX_LEN
    )
    token_ids, seg_ids, is_next = ds[10]
    token_ids
