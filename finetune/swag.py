# References
    # https://rowanzellers.com/swag/
    # https://github.com/rowanz/swagaf/tree/master/data
    # https://github.com/huggingface/transformers/blob/main/examples/legacy/run_swag.py
    # https://huggingface.co/docs/transformers/main/tasks/multiple_choice

# "Given a partial description like 'she opened the hood of the car,' humans can reason about the situation
# and anticipate what might come next ('then, she examined the engine').
# SWAG (Situations With Adversarial Generations) is a large-scale dataset for this task of grounded commonsense inference,
# unifying natural language inference and physically grounded reasoning.
# Each question is a video caption from LSMDC or ActivityNet Captions, with four answer choices
# about what might happen next in the scene. The correct answer is the (real) video caption for the next event in the video;
# the three incorrect answers are adversarially generated and human verified, so as to fool machines but not humans.

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path

from utils import _token_ids_to_segment_ids
from pretrain.wordpiece import load_bert_tokenizer


# "We construct four input sequences, each containing the concatenation of the given sentence (sentence A)
# and a possible continuation (sentence B). The only task-specific parameters introduced is a vector
# whose dot product with the [CLS] token representation $C$ denotes a score for each choice
# which is normalized with a softmax layer."
# We fine-tune the model for 3 epochs with a learning rate of 2e-5 and a batch size of 16.


class SWAGForBERT(Dataset):
    def __init__(self, csv_dir, tokenizer, seq_len, split="train"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.csv_path = Path(csv_dir)/f"{split}.csv"

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")

        self._preprocess()
        self.sent_token_ids = {
            ctx: tokenizer.encode(ctx).ids for ctx in self.raw_data["context"].unique()
        }
        self._get_data()

    def _preprocess(self):
        self.raw_data = pd.read_csv(self.csv_path)
        for col in ["ending0", "ending1", "ending2", "ending3"]:
            self.raw_data[col] = self.raw_data.apply(lambda x: f"""{x["sent2"]} {x[col]}""", axis=1)
        self.raw_data = self.raw_data[["sent1", "ending0", "ending1", "ending2", "ending3", "label"]]
        self.raw_data.rename(
            {
                "sent1": "context",
                "ending0": "example1",
                "ending1": "example2",
                "ending2": "example3",
                "ending3": "example4"
            },
            axis=1,
            inplace=True,
        )
        return self.raw_data

    def _get_data(self):
        self.data = list()
        for row in tqdm(self.raw_data.itertuples(), total=len(self.raw_data)):
            token_ids_batch = list()
            seg_ids_batch = list()
            for i in range(1, 5):
                example_token_ids = self.tokenizer.encode(eval(f"""row.example{i}""")).ids
                token_ids = [self.cls_id] + self.sent_token_ids[row.context] + [self.sep_id] +\
                    example_token_ids
                token_ids = token_ids[: self.seq_len - 1] + [self.sep_id]
                token_ids += [self.pad_id] * (self.seq_len - len(token_ids))
                token_ids = torch.as_tensor(token_ids)

                seg_ids = _token_ids_to_segment_ids(token_ids=token_ids, sep_id=self.sep_id)

                token_ids_batch.append(token_ids)
                seg_ids_batch.append(seg_ids)

            self.data.append((
                torch.stack(token_ids_batch).view(-1, self.seq_len),
                torch.stack(seg_ids_batch).view(-1, self.seq_len),
                torch.as_tensor(row.label),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/bert_from_scratch/pretrain/bookcorpus_vocab.json"
    tokenizer = load_bert_tokenizer(vocab_path)
    MAX_LEN = 128
    csv_dir = "/Users/jongbeomkim/Documents/datasets/swag/"
    ds = SWAGForBERT(csv_dir=csv_dir, tokenizer=tokenizer, seq_len=MAX_LEN)
    BATCH_SIZE = 2
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, (token_ids, seg_ids, label) in enumerate(dl, start=1):
        token_ids.shape, seg_ids.shape, label.shape

    #     token_ids[0, 0, : 40]
    #     token_ids[0, 1, : 40]
