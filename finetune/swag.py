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

import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm.auto import tqdm

from bert.tokenize import prepare_bert_tokenizer
from bert.model import BERTBase, MultipleChoiceHead

pd.options.display.width = sys.maxsize
pd.options.display.max_columns = 10

# "We construct four input sequences, each containing the concatenation of the given sentence (sentence A)
# and a possible continuation (sentence B). The only task-specific parameters introduced is a vector
# whose dot product with the [CLS] token representation $C$ denotes a score for each choice
# which is normalized with a softmax layer."
# We fine-tune the model for 3 epochs with a learning rate of 2e-5 and a batch size of 16.

N_EPOCHS = 3
LR = 2e-5

class SWAGForBERT(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")

        self.corpus = self._preprocess_raw_data(csv_path)
        self.sent2encoded = {ctx: tokenizer.encode(ctx) for ctx in self.corpus["context"].unique()}
        self.data = self._get_data()

    def _preprocess_raw_data(self, csv_path):
        raw_data = pd.read_csv(csv_path)

        for col in ["ending0", "ending1", "ending2", "ending3"]:
            raw_data[col] = raw_data.apply(lambda x: f"""{x["sent2"]} {x[col]}""", axis=1)
        raw_data = raw_data[["sent1", "ending0", "ending1", "ending2", "ending3", "label"]]
        raw_data.rename(
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
        return raw_data

    def _get_data(self):
        data = list()
        for row in tqdm(self.corpus.itertuples(), total=len(self.corpus)):
            batch = list()
            for i in range(1, 5):
                token_ids = [self.cls_id] + self.sent2encoded[row.context].ids + [self.sep_id] +\
                    tokenizer.encode(eval(f"""row.example{i}""")).ids + [self.sep_id]
                token_ids += [self.pad_id] * (self.max_len - len(token_ids))

                batch.append(token_ids)
            data.append({"token_indices": batch, "label": row.label})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.as_tensor(sample["token_indices"]), torch.as_tensor(sample["label"])

if __name__ == "__main__":
    csv_path = "/Users/jongbeomkim/Documents/datasets/swag/train.csv"
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
    MAX_LEN = 512
    swag_ds = SWAGForBERT(csv_path=csv_path, tokenizer=tokenizer, max_len=MAX_LEN)
    BATCH_SIZE = 16
    swag_dl = DataLoader(dataset=swag_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, (token_ids, label) in enumerate(swag_dl, start=1):
        token_ids.shape
        label.shape

    VOCAB_SIZE = 30_522
    model = BERTBase(vocab_size=VOCAB_SIZE)
    token_ids.shape
    token_ids[0].shape
    model(token_ids[0]).shape