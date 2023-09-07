import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import gc
from pathlib import Path
from time import time
from tqdm.auto import tqdm
import argparse

from finetune import config
from utils import get_elapsed_time
from pretrain.wordpiece import load_bert_tokenizer, load_fast_bert_tokenizer
from model import BERTForMultipleChoice
from finetune.swag import SWAGForBERT


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=False)
    parser.add_argument("--pretrained", type=str, required=False)
    parser.add_argument("--batch_size", type=int, required=False, default=16)
    parser.add_argument("--ckpt_path", type=str, required=False)

    args = parser.parse_args()
    return args


def save_checkpoint(step, model, optim, ckpt_path):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "optimizer": optim.state_dict(),
    }
    if config.N_GPUS > 1:
        ckpt["model"] = model.module.state_dict()
    else:
        ckpt["model"] = model.state_dict()
    torch.save(ckpt, str(ckpt_path))


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    gc.collect()
    torch.cuda.empty_cache()

    args = get_args()

    print(f"BATCH_SIZE = {args.batch_size}")
    print(f"N_WORKERS = {config.N_WORKERS}")
    print(f"SEQ_LEN = {config.SEQ_LEN}")

    tokenizer = load_bert_tokenizer(config.VOCAB_PATH)
    # tokenizer = load_fast_bert_tokenizer(vocab_dir=config.VOCAB_DIR)
    train_ds = SWAGForBERT(
        csv_dir=args.csv_dir, tokenizer=tokenizer, seq_len=config.SEQ_LEN, split=args.split,
    )
    train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = BERTForMultipleChoice(
        vocab_size=config.VOCAB_SIZE,
        max_len=config.MAX_LEN,
        pad_id=train_ds.pad_id,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        hidden_size=config.HIDDEN_SIZE,
        mlp_size=config.MLP_SIZE,
        n_choices=config.N_CHOICES,
    ).to(config.DEVICE)
    if config.N_GPUS > 1:
        model = nn.DataParallel(model)

    ### Load pre-trained parameters.
    ckpt = torch.load(args.pretrained, map_location=config.DEVICE)
    if config.N_GPUS > 1:
        model.module.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)

    optim = Adam(model.parameters(), lr=config.LR)

    crit = nn.CrossEntropyLoss(reduction="mean")

    ### Resume
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=config.DEVICE)
        if config.N_GPUS > 1:
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        prev_ckpt_path = Path(args.ckpt_path)
        print(f"Resuming from checkpoint\n    '{args.ckpt_path}'...")
    else:
        step = 0
        prev_ckpt_path = Path(".pth")

    print("Training...")
    start_time = time()
    accum_loss = 0
    step_cnt = 0
    for epoch in range(1, config.N_EPOCHS):
        for step, (token_ids, seg_ids, gt) in enumerate(train_dl, start=1):
            token_ids = token_ids.to(config.DEVICE)
            seg_ids = seg_ids.to(config.DEVICE)
            gt = gt.to(config.DEVICE)

            token_ids = token_ids.view(-1, train_ds.seq_len)
            seg_ids = seg_ids.view(-1, train_ds.seq_len)

            pred = model(token_ids=token_ids, seg_ids=seg_ids)
            loss = crit(pred, gt)

            optim.zero_grad()
            loss.backward()
            optim.step()

            accum_loss += loss.item()
            step_cnt += 1

            if step % (config.N_CKPT_SAMPLES // args.batch_size) == 0:
                print(f"[ {epoch}/{config.N_EPOCHS} ][ {step:,}/{len(train_dl):,} ]", end="")
                print(f"[ {get_elapsed_time(start_time)} ][ NSP loss: {accum_loss / step_cnt:.4f} ]")

                start_time = time()
                accum_loss = 0
                step_cnt = 0

            # cur_ckpt_path = config.CKPT_DIR/f"bookcorpus_step_{step}.pth"
            # save_checkpoint(step=step, model=model, optim=optim, ckpt_path=cur_ckpt_path)
            # if prev_ckpt_path.exists():
            #     prev_ckpt_path.unlink()
            # prev_ckpt_path = cur_ckpt_path
