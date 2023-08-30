from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm

import config
from pretrain.wordpiece import train_bert_tokenizer, load_bert_tokenizer
from pretrain.bookcorpus import BookCorpusForBERT
from model import BERTBaseLM
from pretrain.loss import PretrainingLoss


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
dl = DataLoader(
    ds,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.N_WORKERS,
    pin_memory=True,
    drop_last=True,
)
di = iter(dl)

model = BERTBaseLM(vocab_size=config.VOCAB_SIZE).to(config.DEVICE)

optim = Adam(
    model.parameters(),
    lr=config.MAX_LR,
    betas=(config.BETA1, config.BETA2),
    weight_decay=config.WEIGHT_DECAY,
)

crit = PretrainingLoss()

init_step = 0
running_loss = 0
for step in range(init_step + 1, config.N_STEPS + 1):
    try:
        token_ids, seg_ids, is_next = next(di)
    except StopIteration:
        train_di = iter(dl)
        token_ids, seg_ids, is_next = next(di)
    token_ids = token_ids.to(config.DEVICE)
    seg_ids = seg_ids.to(config.DEVICE)
    is_next = is_next.to(config.DEVICE)

    nsp_pred, mlm_pred = model(seq=token_ids, seg_ids=seg_ids)
    loss = crit(
        mlm_pred=mlm_pred, nsp_pred=nsp_pred, token_ids=token_ids, is_next=is_next,
    )

    optim.zero_grad()
    loss.backward()
    optim.step()

    running_loss += loss.item()
    if (step % config.N_PRINT_STEPS == 0) or (step == config.N_STEPS):
        print(f"""[ {step:,}/{config.N_STEPS:,} ]""", end="")
        print(f"""[ {running_loss / config.N_PRINT_STEPS} ]""")
