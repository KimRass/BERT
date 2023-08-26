# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    # # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html#sec-bert-dataset
    # https://nn.labml.ai/transformers/mlm/index.html

# "For the pre-training corpus we use the BookCorpus (800M words) (Zhu et al., 2015)
# and English Wikipedia (2,500M words)."
# "For Wikipedia we extract only the text passages and ignore lists, tables, and headers.
# It is critical to use a document-level corpus rather than a shuffled sentence-level corpus
# such as the Billion Word Benchmark."

import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from pathlib import Path
from tqdm.auto import tqdm

import config
from wordpiece import encode, build_or_load_vocab

np.set_printoptions(edgeitems=20, linewidth=sys.maxsize, suppress=False)
torch.set_printoptions(edgeitems=16, linewidth=sys.maxsize, sci_mode=True)


def get_bookcorpus_corpus(data_dir):
    print("Generating corpus from BookCorpus dataset...")
    corpus = list()
    for doc_path in tqdm(list(Path(data_dir).glob("**/*.txt"))):
        for parag in open(doc_path, mode="r", encoding="utf-8"):
            parag = parag.strip()
            if parag == "":
                continue

            corpus.append(parag)
    print(f"""Number of corpus: {len(corpus):,}.""")
    return corpus


class BookCorpusForBERT(Dataset):
    def __init__(
        self,
        data_dir,
        vocab,
        max_len,
    ):
        self.data_dir = data_dir
        self.vocab = vocab
        self.max_len = max_len

        # self.cls_id = tokenizer.token_to_id("[CLS]")
        # self.sep_id = tokenizer.token_to_id("[SEP]")
        # self.pad_id = tokenizer.token_to_id("[PAD]")
        # self.unk_id = tokenizer.token_to_id("[UNK]")
        self.cls_id = vocab["[CLS]"]
        self.sep_id = vocab["[SEP]"]
        self.pad_id = vocab["[PAD]"]
        self.unk_id = vocab["[UNK]"]

        self.parags = self._get_parags()
        self.data = self._get_data(self.parags)
    
    def _get_parags(self):
        parags = list()
        for doc_path in tqdm(list(Path(self.data_dir).glob("**/*.txt"))):
            for parag in open(doc_path, mode="r", encoding="utf-8"):
                parag = parag.strip()
                if parag == "":
                    continue

                # token_ids = self.tokenizer.encode(parag).ids
                token_ids = encode(parag, vocab=self.vocab)
                parags.append(
                    {
                        "document": str(doc_path), "paragraph": parag, "token_ids": token_ids
                    }
                )
        return parags

    def _convert_to_bert_input_representation(self, ls_token_ids):
        token_ids = (
            [self.cls_id] + ls_token_ids[0][: self.max_len - 3] + [self.sep_id] + ls_token_ids[1]
        )[: self.max_len - 1] + [self.sep_id]
        token_ids += [self.pad_id] * (self.max_len - len(token_ids))
        return token_ids

    def _get_data(self, parags):
        data = list()

        for id1 in range(len(parags) - 1):
            if random.random() < 0.5:
                is_next = True
                id2 = id1 + 1
            else:
                is_next = False
                id2 = random.randrange(len(parags))
            segs = [parags[id1]["paragraph"], parags[id2]["paragraph"]]
            ls_token_ids = [parags[id1]["token_ids"], parags[id2]["token_ids"]]

            token_ids = self._convert_to_bert_input_representation(ls_token_ids)
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
    corpus = get_bookcorpus_corpus(config.DATA_DIR)
    vocab = build_or_load_vocab(
        corpus=corpus,
        vocab_size=config.VOCAB_SIZE,
        save_path=Path(__file__).parent/"bookcorpus_vocab.json"
    )
    ds = BookCorpusForBERT(data_dir=config.DATA_DIR, vocab=vocab, max_len=config.MAX_LEN)
    # MAX_LEN = 512
    # BATCH_SIZE = 8

    # data_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"
    # vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    # tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
    # tokenizer.encode("[CLS]").tokens
    # dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # for batch, (token_ids, seg_ids, is_next) in enumerate(dl, start=1):
    #   token_ids
