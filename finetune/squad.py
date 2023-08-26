# References
    # https://rajpurkar.github.io/SQuAD-explorer/
    # https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    # https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/
    # https://huggingface.co/learn/nlp-course/chapter7/7?fw=tf

# "SQuAD 1.1, the previous version of the SQuAD dataset,
# contains 100,000+ question-answer pairs on 500+ articles.
# SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions
# written adversarially by crowdworkers to look similar to answerable ones."

import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
from fastapi.encoders import jsonable_encoder
from tqdm.auto import tqdm

from bert.tokenize import prepare_bert_tokenizer
from bert.model import BERTBase, QuestionAnsweringHead

torch.set_printoptions(precision=2, edgeitems=12, linewidth=sys.maxsize, sci_mode=True)

MAX_LEN = 512


class SQuADForBERT(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        max_len=MAX_LEN,
        stride=None,
    ):
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride

        if self.stride is None:
            self.stride = self.max_len // 2
        
        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")
        
        self.corpus = self._get_corpus(json_path)
        self.data = self._get_data(self.corpus)

    def _get_corpus(self, json_path):
        with open(json_path, mode="r") as f:
            raw_data = jsonable_encoder(json.load(f))

        corpus = list()
        error_cnt = 0
        for article in raw_data["data"]:
            for parags in article["paragraphs"]:
                ctx = parags["context"]
                for qa in parags["qas"]:
                    que = qa["question"]
                    ans = qa["answers"]
                    if ans:
                        start_id = ans[0]["answer_start"]
                        end_id = start_id + len(ans[0]["text"])
                        if ctx[start_id: end_id] != ans[0]["text"]:
                            error_cnt += 1
                            continue
                        if "split with " in ans[0]["text"]:
                            start_id, end_id, ctx[start_id: end_id], ctx[start_id: end_id + 5], ans[0]
                    else:
                        # "We treat questions that do not have an answer as having an answer span
                        # with start and end at the [CLS] token."
                        start_id = end_id = 0
                    corpus.append(
                        {"question": que, "context":ctx, "answer": {"start_index": start_id, "end_index": end_id}}
                    )
        print(f"""There were {error_cnt} erro(s).""")
        return corpus

    def _pad_to_maximum_sequence_length(self, token_ids):
        new_token_ids = token_ids + [self.pad_id] * (self.max_len - len(token_ids))
        return new_token_ids

    def _get_segment_indices_from_token_indices(self, token_ids):
        seg_ids = torch.zeros_like(token_ids, dtype=token_ids.dtype, device=token_ids.device)
        is_sep = (token_ids == self.sep_id)
        if is_sep.sum() == 2:
            a, b = is_sep.nonzero()
            seg_ids[a + 1: b + 1] = 1
        return seg_ids

    def _get_data(self, corpus):
        data = list()
        for line in tqdm(corpus):
            que_token_ids = tokenizer.encode(line["question"]).ids
            ctx_encoded = tokenizer.encode(line["context"])
            ctx_token_ids = ctx_encoded.ids

            len_que = len(que_token_ids)
            len_ctx = len(ctx_token_ids)

            if (line["answer"]["start_index"], line["answer"]["end_index"]) == (0, 0):
                start_id, end_id = 0, 0
            else:
                ctx_offsets = ctx_encoded.offsets
                for id_, (s, e) in enumerate(ctx_offsets):
                    if s <= line["answer"]["start_index"] < e:
                        start_id = id_ + len_que + 2
                    if s < line["answer"]["end_index"] <= e:
                        end_id = id_ + len_que + 3

            if len_que + len_ctx + 3 > self.max_len:
                for i in range(0, len_ctx // 2, self.stride):
                    # "We represent the input question and passage as a single packed sequence,
                    # with the question using the A embedding and the passage using the B embedding."
                    token_ids = [self.cls_id] + que_token_ids + [self.sep_id] +\
                        ctx_token_ids[i: i + self.max_len - len_que - 3] + [self.sep_id]

                    if (start_id, end_id) != (0, 0):
                        start_id -= i
                        end_id -= i

                    token_ids = self._pad_to_maximum_sequence_length(token_ids)
                    token_ids = torch.as_tensor(token_ids)
                    seg_ids = self._get_segment_indices_from_token_indices(token_ids)
                    start_id = torch.as_tensor(start_id)
                    end_id = torch.as_tensor(end_id)

                    data.append(
                        {
                            "token_indices": token_ids,
                            "segment_indices": seg_ids,
                            "start_index": start_id,
                            "end_index": end_id
                        }
                    )
            else:
                token_ids = [self.cls_id] + que_token_ids + [self.sep_id] + ctx_token_ids + [self.sep_id]

                token_ids = self._pad_to_maximum_sequence_length(token_ids)
                token_ids = torch.as_tensor(token_ids)
                seg_ids = self._get_segment_indices_from_token_indices(token_ids)
                start_id = torch.as_tensor(start_id)
                end_id = torch.as_tensor(end_id)

                data.append(
                    {
                        "token_indices": token_ids,
                        "segment_indices": seg_ids,
                        "start_index": start_id,
                        "end_index": end_id
                    }
                )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            sample["token_indices"], sample["segment_indices"], sample["start_index"], sample["end_index"],
        )


if __name__ == "__main__":
    VOCAB_SIZE = 30_522
    model = BERTBase(vocab_size=VOCAB_SIZE)
    qa_head = QuestionAnsweringHead(hidden_dim=768)
    # x = torch.randn((8, 512, 768))

    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
    json_path = "/Users/jongbeomkim/Documents/datasets/train-v2.0.json"
    squad_ds = SQuADForBERT(json_path=json_path, tokenizer=tokenizer, max_len=MAX_LEN)
    BATCH_SIZE = 8
    squad_dl = DataLoader(dataset=squad_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, (token_ids, seg_ids, gt_start_id, gt_end_id) in enumerate(squad_dl, start=1):
        logit = model(token_ids, seg_ids=seg_ids)
        pred_start_id, pred_end_id = qa_head(logit)
        pred_start_id, gt_start_id
        pred_end_id, gt_end_id


    # The probability space for the start and end answer span positions is extended to include the position of the [CLS] token. For prediction, we compare the score of the no-answer span: snull = S C + E C
    # to the score of the best non-null span s^i;j = maxj iS Ti + E Tj . We predict a non-null answer
    # when s^i;j > snull +   , where the threshold is selected on the dev set to maximize F1.

    # The training objective is the sum of the log-likelihoods of the correct start and end positions.
    # We fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32. Table 2 shows top leaderboard entries as well