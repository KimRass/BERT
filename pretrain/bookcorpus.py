# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    # https://nn.labml.ai/transformers/mlm/index.html
    # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html

# "For the pre-training corpus we use the BookCorpus (800M words) (Zhu et al., 2015)
# and English Wikipedia (2,500M words)."
# "For Wikipedia we extract only the text passages and ignore lists, tables, and headers.
# It is critical to use a document-level corpus rather than a shuffled sentence-level corpus
# such as the Billion Word Benchmark."

import sys

sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/bert_from_scratch")

import torch
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
from tqdm.auto import tqdm

import config
from pretrain.wordpiece import train_bert_tokenizer, load_bert_tokenizer


# def _get_parags(epubtxt_dir):
#     parags = list()
#     for doc_path in tqdm(list(Path(epubtxt_dir).glob("*.txt"))):
#         for parag in open(doc_path, mode="r", encoding="utf-8"):
#             parag = parag.strip()
#             if parag == "":
#                 continue
#             # if "テ" in parag:
#             #     print(doc_path)
#             token_ids = tokenizer.encode(parag).ids
#             parags.append(
#                 {
#                     "document": str(doc_path), "paragraph": parag, "token_ids": token_ids,
#                 }
#             )
#     return parags


class BookCorpusForBERT(Dataset):
    def __init__(
        self,
        epubtxt_dir,
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

        self.parags = self._get_parags()
        self.data = self._get_data(self.parags)
    
    def _get_parags(self):
        parags = list()
        for doc_path in tqdm(list(Path(self.epubtxt_dir).glob("*.txt"))):
            for parag in open(doc_path, mode="r", encoding="utf-8"):
                parag = parag.strip()
                if parag == "":
                    continue

                token_ids = self.tokenizer.encode(parag).ids
                parags.append(
                    {
                        "document": str(doc_path), "paragraph": parag, "token_ids": token_ids,
                    }
                )
        return parags

    def _to_bert_input(self, ls_token_ids):
        token_ids = (
            [self.cls_id] + ls_token_ids[0][: self.max_len - 3] + [self.sep_id] + ls_token_ids[1]
        )[: self.max_len - 1] + [self.sep_id]
        token_ids += [self.pad_id] * (self.max_len - len(token_ids))
        return token_ids

    def _get_data(self, parags):
        data = list()

        for idx1 in tqdm(range(len(parags) - 1)):
            if random.random() < 0.5:
                is_next = True
                idx2 = idx1 + 1
            else:
                is_next = False
                idx2 = random.randrange(len(parags))

            parag1 = parags[idx1]["paragraph"]
            parag2 = parags[idx2]["paragraph"]
            segs = [parag1, parag2]

            token_ids1 = parags[idx1]["token_ids"]
            token_ids2 = parags[idx2]["token_ids"]
            ls_token_ids = [token_ids1, token_ids2]

            token_ids = self._to_bert_input(ls_token_ids)
            data.append(
                {
                    "segments": segs,
                    "lists_of_token_ids": ls_token_ids,
                    "token_ids": token_ids,
                    "is_next": is_next
                }
            )
        return data

    def _get_segment_ids_from_token_ids(self, token_ids):
        seg_ids = torch.zeros_like(token_ids, dtype=token_ids.dtype, device=token_ids.device)
        is_sep = (token_ids == self.sep_id)
        if is_sep.sum() == 2:
            a, b = is_sep.nonzero()
            seg_ids[a + 1: b + 1] = 1
        return seg_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = torch.as_tensor(self.data[idx]["token_ids"])
        seg_ids = self._get_segment_ids_from_token_ids(token_ids)
        return token_ids, seg_ids, torch.as_tensor(self.data[idx]["is_next"])


if __name__ == "__main__":
    corpus_files = list(Path(config.EPUBTXT_DIR).glob("*.txt"))
    if not Path(config.VOCAB_PATH).exists():
        train_bert_tokenizer(
            vocab_size=config.VOCAB_SIZE,
            vocab_path=config.VOCAB_PATH,
            min_freq=config.MIN_FREQ,
            corpus_files=corpus_files,
            post_processor=True,
        )
    tokenizer = load_bert_tokenizer(config.VOCAB_PATH)

    ds = BookCorpusForBERT(
        epubtxt_dir=config.EPUBTXT_DIR, tokenizer=tokenizer, max_len=config.MAX_LEN
    )
    ds[10]
