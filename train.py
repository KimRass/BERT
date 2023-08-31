import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from pathlib import Path
import argparse
from time import time

import config
from pretrain.wordpiece import train_bert_tokenizer, load_bert_tokenizer
from pretrain.bookcorpus import BookCorpusForBERT
from model import BERTBaseLM
from pretrain.loss import PretrainingLoss
from utils import get_elapsed_time


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


def save_checkpoint(step, model, optim, scaler, ckpt_path):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if config.N_GPUS > 1:
        ckpt["model"] = model.module.state_dict()
    else:
        ckpt["model"] = model.state_dict()

    torch.save(ckpt, str(ckpt_path))


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    args = get_args()

    BATCH_SIZE = args.batch_size if args.batch_size != 0 else config.BATCH_SIZE
    N_STEPS = (256 * 512 * 1_000_000) // (BATCH_SIZE * config.MAX_LEN)
    print(f"""BATCH_SIZE = {BATCH_SIZE}""")
    print(f"""MAX_LEN = {config.MAX_LEN}""")
    print(f"""N_STEPS = {N_STEPS}""")

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
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    di = iter(dl)

    model = BERTBaseLM(vocab_size=config.VOCAB_SIZE).to(config.DEVICE)
    if config.N_GPUS > 1:
        model = nn.DataParallel(model)

    optim = Adam(
        model.parameters(),
        lr=config.MAX_LR,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY,
    )

    scaler = GradScaler(enabled=True if config.AUTOCAST else False)

    crit = PretrainingLoss()

    init_step = 0
    running_loss = 0
    step_cnt = 0
    start_time = time()
    for step in range(init_step + 1, N_STEPS + 1):
        try:
            token_ids, seg_ids, is_next = next(di)
        except StopIteration:
            train_di = iter(dl)
            token_ids, seg_ids, is_next = next(di)
        token_ids = token_ids.to(config.DEVICE)
        seg_ids = seg_ids.to(config.DEVICE)
        is_next = is_next.to(config.DEVICE)

        with torch.autocast(
            device_type=config.DEVICE.type,
            dtype=torch.float16,
            enabled=True if config.AUTOCAST else False,
        ):
            nsp_pred, mlm_pred = model(seq=token_ids, seg_ids=seg_ids)
            loss = crit(
                mlm_pred=mlm_pred, nsp_pred=nsp_pred, token_ids=token_ids, is_next=is_next,
            )
        optim.zero_grad()
        if config.AUTOCAST:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        running_loss += loss.item()
        step_cnt += 1

        if (step % config.N_PRINT_STEPS == 0) or (step == N_STEPS):
            print(f"""[ {step:,}/{N_STEPS:,} ][ {get_elapsed_time(start_time)} ]""", end="")
            print(f"""[ {running_loss / step_cnt:.3f} ]""")

            running_loss = 0

        ckpt_path = config.CKPT_DIR/f"""step_{step}.pth"""
        if (step % config.N_PRINT_STEPS == 0) or (step == N_STEPS):
            save_checkpoint(
                step=step, model=model, optim=optim, scaler=scaler, ckpt_path=ckpt_path
            )
