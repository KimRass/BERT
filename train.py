import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import gc
from pathlib import Path
from time import time

import config
from pretrain.wordpiece import load_bert_tokenizer
from pretrain.bookcorpus import BookCorpusForBERT
from pretrain.model import BERTForPretraining
from pretrain.masked_language_model import MaskedLanguageModel
from pretrain.loss import LossForPretraining
from utils import get_args, get_elapsed_time
from pretrain.evalute import get_nsp_acc, get_mlm_acc

# torch.set_printoptions(sci_mode=False)
# torch.set_printoptions(linewidth=180)


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

    # "We train with batch size of 256 sequences (256 sequences * 512 tokens
    # = 128,000 tokens/batch) for 1,000,000 steps, which is approximately 40 epochs
    # over the 3.3 billion word corpus." (Comment: 256 * 512 * 1,000,000 / 3,300,000,000
    # = 39.7)
    # 학습이 너무 오래 걸리므로 절반 만큼만 학습하겠습니다.
    N_STEPS = (256 * 512 * 500_000) // (args.batch_size * config.SEQ_LEN)
    print(f"""N_WORKERS = {config.N_WORKERS}""")
    print(f"""MAX_LEN = {config.MAX_LEN}""")
    print(f"""SEQ_LEN = {config.SEQ_LEN}""")
    print(f"""N_STEPS = {N_STEPS:,}""", end="\n\n")

    tokenizer = load_bert_tokenizer(config.VOCAB_PATH)
    ds = BookCorpusForBERT(
        epubtxt_dir=args.epubtxt_dir,
        tokenizer=tokenizer,
        seq_len=config.SEQ_LEN,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    di = iter(dl)

    model = BERTForPretraining( # Smaller than BERT-Base
        vocab_size=config.VOCAB_SIZE,
        max_len=config.MAX_LEN,
        pad_id=ds.pad_id,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        hidden_size=config.HIDDEN_SIZE,
        mlp_size=config.MLP_SIZE,
    ).to(config.DEVICE)
    if config.N_GPUS > 1:
        model = nn.DataParallel(model)
    no_mask_token_ids = [ds.cls_id, ds.sep_id, ds.pad_id, ds.unk_id]
    mlm = MaskedLanguageModel(
        vocab_size=config.VOCAB_SIZE,
        mask_id=tokenizer.token_to_id("[MASK]"),
        no_mask_token_ids=no_mask_token_ids,
        select_prob=config.SELECT_PROB,
        mask_prob=config.MASK_PROB,
        randomize_prob=config.RANDOMIZE_PROB,
    )

    optim = Adam(
        model.parameters(),
        lr=config.MAX_LR,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY,
    )

    crit = LossForPretraining()

    ### Resume
    if config.CKPT_PATH is not None:
        ckpt = torch.load(config.CKPT_PATH, map_location=config.DEVICE)
        if config.N_GPUS > 1:
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        init_step = ckpt["step"]
        prev_ckpt_path = config.CKPT_PATH
        print(f"""Resuming from checkpoint\n    '{config.CKPT_PATH}'...""")
    else:
        init_step = 0
        prev_ckpt_path = ".pth"

    print("Training...")
    start_time = time()
    accum_nsp_loss = 0
    accum_mlm_loss = 0
    accum_nsp_acc = 0
    accum_mlm_acc = 0
    step_cnt = 0
    for step in range(init_step + 1, N_STEPS + 1):
        try:
            gt_token_ids, seg_ids, gt_is_next = next(di)
        except StopIteration:
            di = iter(dl)
            gt_token_ids, seg_ids, gt_is_next = next(di)

        gt_token_ids = gt_token_ids.to(config.DEVICE)
        seg_ids = seg_ids.to(config.DEVICE)
        gt_is_next = gt_is_next.to(config.DEVICE)

        # masked_token_ids = mlm(gt_token_ids)

        # pred_is_next = model(token_ids=masked_token_ids, seg_ids=seg_ids)
        pred_is_next = model(token_ids=gt_token_ids, seg_ids=seg_ids)
        # print(pred_is_next)
        nsp_loss = crit(
            pred_is_next=pred_is_next,
            gt_is_next=gt_is_next,
            # pred_token_ids=pred_token_ids,
            # gt_token_ids=gt_token_ids,
        )
        # loss = nsp_loss + mlm_loss

        optim.zero_grad()
        # loss.backward()
        nsp_loss.backward()
        optim.step()

        accum_nsp_loss += nsp_loss.item()
        # accum_mlm_loss += mlm_loss.item()
        nsp_acc = get_nsp_acc(pred_is_next=pred_is_next, gt_is_next=gt_is_next)
        # mlm_acc = get_mlm_acc(pred_token_ids=pred_token_ids, gt_token_ids=gt_token_ids)
        accum_nsp_acc += nsp_acc
        # accum_mlm_acc += mlm_acc
        step_cnt += 1

        if (step % (config.N_CKPT_SAMPLES // args.batch_size) == 0) or (step == N_STEPS):
            print(torch.argmax(pred_is_next, dim=-1))
            print(gt_is_next, end="\n\n")
            print(f"""[ {step:,}/{N_STEPS:,} ][ {get_elapsed_time(start_time)} ]""", end="")
            print(f"""[ NSP loss: {accum_nsp_loss / step_cnt:.4f} ]""", end="")
            # print(f"""[ MLM loss: {accum_mlm_loss / step_cnt:.4f} ]""", end="")
            print(f"""[ NSP acc: {accum_nsp_acc / step_cnt:.3f} ]""")
            # print(f"""[ MLM acc: {accum_mlm_acc / step_cnt:.3f} ]""")

            start_time = time()
            accum_nsp_loss = 0
            accum_mlm_loss = 0
            accum_nsp_acc = 0
            accum_mlm_acc = 0
            step_cnt = 0

            cur_ckpt_path = config.CKPT_DIR/f"""bookcorpus_step_{step}.pth"""
            save_checkpoint(step=step, model=model, optim=optim, ckpt_path=cur_ckpt_path)
            if Path(prev_ckpt_path).exists():
                prev_ckpt_path.unlink()
            prev_ckpt_path = cur_ckpt_path
