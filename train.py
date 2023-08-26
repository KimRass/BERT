from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from wordpiece import build_vocabulary, tokenize, encode
from pretrain.bookcorpus import BookCorpusForBERT
from model import (
    BERTBase,
    MaskedLanguageModelHead,
    NextSentencePredictionHead
)
from pretrain.loss import PretrainingLoss

# VOCAB_SIZE = 30_522
VOCAB_SIZE = 30_522 // 10

data_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"
corpus = list()
for doc_path in tqdm(list(Path(data_dir).glob("**/*.txt"))):
    for parag in open(doc_path, mode="r", encoding="utf-8"):
        parag = parag.strip()
        if parag == "":
            continue
        corpus.append(parag)
len(corpus)
build_vocabulary(
    corpus=corpus, vocab_size=VOCAB_SIZE, save_path="/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab.json"
)



MAX_LEN = 512
BATCH_SIZE = 8
N_STEPS = 1_000_000
N_WARMUP_STEPS = 10_000
MAX_LR = 1e-4

vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
# tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
ds = BookCorpusForBERT(data_dir=data_dir, vocab=vocab, max_len=MAX_LEN)
dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
model = BERTBase(vocab_size=VOCAB_SIZE)
mlm_head = MaskedLanguageModelHead(vocab_size=VOCAB_SIZE)
nsp_head = NextSentencePredictionHead()
criterion = PretrainingLoss()

for batch, (token_ids, seg_ids, is_next) in enumerate(dl, start=1):
    optimizer.zero_grad()

    bert_out = model(seq=token_ids, seg_label=seg_ids)
    mlm_logit = mlm_head(bert_out)
    nsp_logit = nsp_head(bert_out)
    loss = criterion(mlm_logit=mlm_logit, nsp_logit=nsp_logit, label=is_next)

    loss.backward()
    optimizer.step()

    running_loss += loss.item()