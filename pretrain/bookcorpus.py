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
from pathlib import Path
import pysbd
import re

from utils import _token_ids_to_segment_ids, REGEX
from pretrain.wordpiece import parse

os.environ["TOKENIZERS_PARALLELISM"] = "true"


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
        ### Add '[CLS]' and '[SEP]' tokens.
        token_ids = [self.cls_id] + former_token_ids[: self.seq_len - 3] + [self.sep_id]\
            + latter_token_ids
        token_ids = token_ids[: self.seq_len - 1] + [self.sep_id]
        ### Pad.
        token_ids += [self.pad_id] * (self.seq_len - len(token_ids))
        return torch.as_tensor(token_ids)

    def _encode(self, x):
        encoding = self.tokenizer(
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

    def __len__(self):
        return len(self.lines) - 1

    def __getitem__(self, idx):
        former_line = self.lines[idx]
        former_token_ids = self._encode(former_line)
        latter_line, is_next = self._sample_latter_sentence(idx)
        latter_token_ids = self._encode(latter_line)

        token_ids = self._to_bert_input(
            former_token_ids=former_token_ids, latter_token_ids=latter_token_ids,
        )
        seg_ids = _token_ids_to_segment_ids(token_ids=token_ids, sep_id=self.sep_id)
        return token_ids, seg_ids, is_next


class BookCorpusForRoBERTa(Dataset):
    def __init__(
        self,
        epubtxt_dir,
        tokenizer,
        seq_len,
        mode="full_sentences",
    ):
        self.epubtxt_dir = epubtxt_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mode = mode

        self.segmentor = pysbd.Segmenter(language="en", clean=False)

        self.unk_id = tokenizer.unk_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        if mode == "full_sentences":
            self._parse(perform_sbd=True)
        self._get_data()

    def _disambiguate_sentence_boundary(self, text):
        segmented = self.segmentor.segment(text)
        return [i.strip() for i in segmented]
    
    def _parse(self, perform_sbd):
        print("Parsing BookCorpus...")
        self.lines = list()
        for doc_path in tqdm(list(Path(self.epubtxt_dir).glob("*.txt"))):
            with open(doc_path, mode="r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if (not line) or (re.search(pattern=REGEX, string=line)) or (line.count(" ") < 1):
                        continue

                if perform_sbd:
                    sents = self._disambiguate_sentence_boundary(line)
                    for sent in sents:
                        token_ids = self.tokenizer.encode(
                            sent, truncation=True, max_length=self.seq_len,
                        )[1: -1]
                        self.lines.append(
                            {
                                "document": str(doc_path),
                                "paragraph": line,
                                "sentence": sent,
                                "token_ids": token_ids
                            }
                        )
                else:
                    token_ids = self.tokenizer.encode(
                        line, truncation=True, max_length=self.seq_len,
                    )[1: -1]
                    self.lines.append(
                        {
                            "document": str(doc_path),
                            "paragraph": line,
                            "token_ids": token_ids
                        }
                    )

    def _to_bert_input(self, token_ids_ls):
        token_ids = sum(token_ids_ls, list())
        token_ids = token_ids[: self.seq_len - 2]
        token_ids = [self.cls_id] + token_ids + [self.sep_id]
        token_ids += [self.pad_id] * (self.seq_len - len(token_ids))
        return token_ids

    def _get_data(self):
        self.data = list()
        if self.mode == "full_sentences":
            sents = [self.lines[0]["sentence"]]
            token_ids_ls = [self.lines[0]["token_ids"]]
            for id_ in range(1, len(self.lines)):
                # "Inputs may cross document boundaries. When we reach the end of one document,
                # we begin sampling sentences from the next document
                # and add an extra separator token between documents."
                if self.lines[id_ - 1]["document"] != self.lines[id_]["document"]:
                    token_ids_ls.append([self.sep_id])

                # Each input is packed with full sentences sampled contiguously
                # from one or more documents, such that the total length is at most 512 tokens.
                if len(sum(token_ids_ls, list())) + len(self.lines[id_]["token_ids"]) > self.seq_len - 2 or\
                    id_ == len(self.lines) - 1:
                    token_ids = self._to_bert_input(token_ids_ls)
                    self.data.append(
                        {"sentences": sents, "lists_of_token_ids": token_ids_ls, "token_ids": token_ids}
                    )

                    sents = list()
                    token_ids_ls = list()
                sents.append(self.lines[id_]["sentence"])
                token_ids_ls.append(self.lines[id_]["token_ids"])
        return self.datadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.as_tensor(self.data[idx]["token_ids"])
