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
from finetune.evaluate import TopKAccuracy


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_dir", type=str, required=True)
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


@torch.no_grad()
def validate(val_dl, model, metric):
    print(f"""Validating...""")
    model.eval()
    sum_acc = 0
    for token_ids, seg_ids, gt in val_dl:
        token_ids = token_ids.to(config.DEVICE)
        seg_ids = seg_ids.to(config.DEVICE)
        gt = gt.to(config.DEVICE)

        token_ids = token_ids.view(-1, config.SEQ_LEN)
        seg_ids = seg_ids.view(-1, config.SEQ_LEN)

        pred = model(token_ids=token_ids, seg_ids=seg_ids)
        acc = metric(pred=pred, gt=gt)
        sum_acc += acc
    avg_acc = sum_acc / len(val_dl)
    print(f"""Average accuracy: {avg_acc:.3f}""")

    model.train()
    return avg_acc


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    gc.collect()
    torch.cuda.empty_cache()

    args = get_args()

    print(f"BATCH_SIZE = {args.batch_size}")
    print(f"N_WORKERS = {config.N_WORKERS}")
    print(f"SEQ_LEN = {config.SEQ_LEN}")

    tokenizer = load_bert_tokenizer(config.VOCAB_PATH)
    train_ds = SWAGForBERT(
        csv_dir=args.csv_dir, tokenizer=tokenizer, seq_len=config.SEQ_LEN, split="train",
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_ds = SWAGForBERT(
        csv_dir=args.csv_dir, tokenizer=tokenizer, seq_len=config.SEQ_LEN, split="val",
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

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
    print(f"Loaded pre-trained parameters from\n    '{str(Path(args.pretrained).name)}.'")

    optim = Adam(model.parameters(), lr=config.LR)

    crit = nn.CrossEntropyLoss(reduction="mean")
    metric = TopKAccuracy(k=1) 

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

            token_ids = token_ids.view(-1, config.SEQ_LEN)
            seg_ids = seg_ids.view(-1, config.SEQ_LEN)

            pred = model(token_ids=token_ids, seg_ids=seg_ids)
            loss = crit(pred, gt)

            optim.zero_grad()
            loss.backward()
            optim.step()

            accum_loss += loss.item()
            step_cnt += 1

            if step % (config.N_CKPT_SAMPLES // args.batch_size) == 0:
                print(f"[ {epoch}/{config.N_EPOCHS} ][ {step:,}/{len(train_dl):,} ]", end="")
                print(f"[ {get_elapsed_time(start_time)} ][ Loss: {accum_loss / step_cnt:.4f} ]")

                start_time = time()
                accum_loss = 0
                step_cnt = 0

                avg_acc = validate(val_dl=val_dl, model=model, metric=metric)
            # cur_ckpt_path = config.CKPT_DIR/f"bookcorpus_step_{step}.pth"
            # save_checkpoint(step=step, model=model, optim=optim, ckpt_path=cur_ckpt_path)
            # if prev_ckpt_path.exists():
            #     prev_ckpt_path.unlink()
            # prev_ckpt_path = cur_ckpt_path
