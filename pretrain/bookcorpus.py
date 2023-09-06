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
from typing import Literal
import pysbd

from pretrain.wordpiece import parse

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class BookCorpusForBERT(Dataset):
    def __init__(
        self,
        epubtxt_dir,
        tokenizer,
        seq_len,
        tokenize_in_advance=False,
        chunk_size=2 ** 4,
    ):
        self.epubtxt_dir = epubtxt_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokenize_in_advance = tokenize_in_advance
        self.chunk_size = chunk_size

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")
        # self.cls_id = tokenizer.cls_token_id
        # self.sep_id = tokenizer.sep_token_id
        # self.pad_id = tokenizer.pad_token_id
        # self.unk_id = tokenizer.unk_token_id

        self.lines = parse(epubtxt_dir)
        if tokenize_in_advance:
            self._tokenize()

    def _tokenize(self):
        print("Tokenizing BookCorpus...")
        self.token_ids_ls = list()
        for idx in tqdm(range(0, len(self.lines), self.chunk_size)):
            encoding = self.tokenizer(self.lines[idx: idx + self.chunk_size])
            self.token_ids_ls.extend([i[1: -1] for i in encoding["input_ids"]])
    #         encoded = self.tokenizer.encode_batch(self.lines[idx: idx + self.chunk_size])
    #         self.token_ids_ls.extend([i.ids for i in encoded])
        print("Completed")

    def _to_bert_input(self, former_token_ids, latter_token_ids):
        ### Add '[CLS]' and '[SEP]' tokens.
        token_ids = [self.cls_id] + former_token_ids[: self.seq_len - 3] + [self.sep_id] + latter_token_ids
        token_ids = token_ids[: self.seq_len - 1] + [self.sep_id]
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
        if self.tokenize_in_advance:
            latter_token_ids = self.token_ids_ls[latter_idx]
            return latter_token_ids, torch.as_tensor(is_next)
        else:
            latter_line = self.lines[latter_idx]
            return latter_line, torch.as_tensor(is_next)

    def _token_ids_to_segment_ids(self, token_ids):
        seg_ids = torch.zeros_like(token_ids, dtype=token_ids.dtype, device=token_ids.device)
        is_sep = (token_ids == self.sep_id)
        # if is_sep.sum() == 2:
        first_sep, second_sep = is_sep.nonzero()
        # The positions from right after the first '[SEP]' token and to the second '[SEP]' token
        seg_ids[first_sep + 1: second_sep + 1] = 1
        return seg_ids

    def __len__(self):
        if self.tokenize_in_advance:
            return len(self.token_ids_ls) - 1
        else:
            return len(self.lines) - 1

    def __getitem__(self, idx):
        if self.tokenize_in_advance:
            former_token_ids = self.token_ids_ls[idx]
            latter_token_ids, is_next = self._sample_latter_sentence(idx)
        else:
            former_line = self.lines[idx]
            former_token_ids = self.tokenizer.encode(former_line).ids
            latter_line, is_next = self._sample_latter_sentence(idx)
            latter_token_ids = self.tokenizer.encode(latter_line).ids

        token_ids = self._to_bert_input(
            former_token_ids=former_token_ids, latter_token_ids=latter_token_ids,
        )
        seg_ids = self._token_ids_to_segment_ids(token_ids)
        return token_ids, seg_ids, is_next


class BookCorpusForRoBERTa(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer,
        max_len,
        mode: Literal["segment_pair", "sentence_pair", "full_sentences", "doc_sentences"]="doc_sentences"
    ):
        assert mode in ["segment_pair", "sentence_pair", "full_sentences", "doc_sentences"],\
            "The argument `mode` should be one of `'segment_pair'`, `'sentence_pair'`, 'full_sentences', and 'doc_sentences'."

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        self.segmentor = pysbd.Segmenter(language="en", clean=False)

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")

        if mode == "segment_pair":
            self.corpus = self._get_corpus(perform_sbd=False)
        else:
            self.corpus = self._get_corpus(perform_sbd=True)
        self.data = self._get_data(self.corpus)

    def _disambiguate_sentence_boundary(self, text):
        segmented = self.segmentor.segment(text)
        return [i.strip() for i in segmented]
    
    def _get_corpus(self, perform_sbd):
        corpus = list()
        for doc_path in tqdm(list(Path(data_dir).glob("**/*.txt"))):
            for parag in open(doc_path, mode="r", encoding="utf-8"):
                parag = parag.strip()
                if parag == "":
                    continue

                if perform_sbd:
                    sents = self._disambiguate_sentence_boundary(parag)
                    for sent in sents:
                        token_ids = self.tokenizer.encode(sent).ids
                        corpus.append(
                            {
                                "document": str(doc_path),
                                "paragraph": parag,
                                "sentence": sent,
                                "token_indices": token_ids
                            }
                        )
                else:
                    token_ids = self.tokenizer.encode(parag).ids
                    corpus.append(
                        {
                            "document": str(doc_path),
                            "paragraph": parag,
                            "token_indices": token_ids
                        }
                    )
        return corpus

    def _convert_to_bert_input_representation(self, ls_token_ids):
        if self.mode in ["segment_pair", "sentence_pair"]:
            token_ids = (
                [self.cls_id] + ls_token_ids[0][: self.max_len - 3] + [self.sep_id] + ls_token_ids[1]
            )[: self.max_len - 1] + [self.sep_id]
            token_ids += [self.pad_id] * (self.max_len - len(token_ids))
        else:
            token_ids = sum(ls_token_ids, list())
            token_ids = token_ids[: self.max_len - 2]
            token_ids = [self.cls_id] + token_ids + [self.sep_id]
            token_ids += [self.pad_id] * (self.max_len - len(token_ids))
        return token_ids

    def _get_data(self, corpus):
        data = list()

        # "Each input has a pair of segments, which can each contain multiple natural sentences,
        # but the total combined length must be less than 512 tokens."
        if self.mode == "segment_pair":
            for id1 in range(len(corpus) - 1):
                if random.random() < 0.5:
                    is_next = True
                    id2 = id1 + 1
                else:
                    is_next = False
                    id2 = random.randrange(len(corpus))
                segs = [corpus[id1]["paragraph"], corpus[id2]["paragraph"]]
                ls_token_ids = [corpus[id1]["token_indices"], corpus[id2]["token_indices"]]

                token_ids = self._convert_to_bert_input_representation(ls_token_ids)
                data.append(
                    {
                        "segments": segs,
                        "lists_of_token_indices": ls_token_ids,
                        "token_indices": token_ids,
                        "is_next": is_next
                    }
                )

        # Each input contains a pair of natural sentences,
        # either sampled from a contiguous portion of one document or from separate documents.
        # Since these inputs are significantly shorter than 512 tokens,
        # we increase the batch size so that the total number of tokens remains similar to "SEGMENT-PAIR" + NSP.
        elif self.mode == "sentence_pair":
            for id1 in range(len(corpus) - 1):
                if random.random() < 0.5:
                    is_next = True
                    id2 = id1 + 1
                else:
                    is_next = False
                    id2 = random.randrange(len(corpus))
                sents = [corpus[id1]["sentence"], corpus[id2]["sentence"]]
                ls_token_ids = [corpus[id1]["token_indices"], corpus[id2]["token_indices"]]

                token_ids = self._convert_to_bert_input_representation(ls_token_ids)
                data.append(
                    {
                        "sentences": sents,
                        "lists_of_token_indices": ls_token_ids,
                        "token_indices": token_ids,
                        "is_next": is_next
                    }
                )

        elif self.mode == "full_sentences":
            sents = [corpus[0]["sentence"]]
            ls_token_ids = [corpus[0]["token_indices"]]
            for id_ in range(1, len(corpus)):
                # "Inputs may cross document boundaries. When we reach the end of one document,
                # we begin sampling sentences from the next document
                # and add an extra separator token between documents."
                if corpus[id_ - 1]["document"] != corpus[id_]["document"]:
                    ls_token_ids.append([self.sep_id])

                # Each input is packed with full sentences sampled contiguously
                # from one or more documents, such that the total length is at most 512 tokens.
                if len(sum(ls_token_ids, list())) + len(corpus[id_]["token_indices"]) > self.max_len - 2 or\
                id_ == len(corpus) - 1:
                    token_ids = self._convert_to_bert_input_representation(ls_token_ids)
                    data.append(
                        {"sentences": sents, "lists_of_token_indices": ls_token_ids, "token_indices": token_ids}
                    )

                    sents = list()
                    ls_token_ids = list()
                sents.append(corpus[id_]["sentence"])
                ls_token_ids.append(corpus[id_]["token_indices"])

        # "Inputs sampled near the end of a document may be shorter than 512 tokens,
        # so we dynamically increase the batch size in these cases
        # to achieve a similar number of total tokens as 'FULL-SENTENCES'."
        else: # `"doc_sentences"`
            sents = [corpus[0]["sentence"]]
            ls_token_ids = [corpus[0]["token_indices"]]
            for id_ in range(1, len(corpus)):
                # "except that they may not cross document boundaries.
                # Inputs are constructed similarly to 'FULL-SENTENCES'",
                if corpus[id_ - 1]["document"] != corpus[id_]["document"] or\
                len(sum(ls_token_ids, list())) + len(corpus[id_]["token_indices"]) > self.max_len - 2 or\
                id_ == len(corpus) - 1:
                    token_ids = self._convert_to_bert_input_representation(ls_token_ids)
                    data.append(
                        {"sentences": sents, "lists_of_token_indices": ls_token_ids, "token_indices": token_ids}
                    )

                    sents = list()
                    ls_token_ids = list()
                sents.append(corpus[id_]["sentence"])
                ls_token_ids.append(corpus[id_]["token_indices"])
        return data

    def _get_segment_indices_from_token_indices(self, token_ids):
        seg_ids = torch.zeros_like(token_ids, dtype=token_ids.dtype, device=token_ids.device)
        is_sep = (token_ids == self.sep_id)
        if is_sep.sum() == 2:
            a, b = is_sep.nonzero()
            seg_ids[a + 1: b + 1] = 1
        return seg_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode in ["segment_pair", "sentence_pair"]:
            token_ids = torch.as_tensor(self.data[idx]["token_indices"])
            seg_ids = self._get_segment_indices_from_token_indices(token_ids)
            return token_ids, seg_ids, torch.as_tensor(self.data[idx]["is_next"])
        else:
            return torch.as_tensor(self.data[idx]["token_indices"])
