# import sys

# sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/bert_from_scratch")

import torch
from pathlib import Path

### Data
VOCAB_SIZE = 30_522 // 2 # 학습이 너무 오래 걸리므로 절반으로 줄이겠습니다.
# "To speed up pretraing in our experiments, we pre-train the model
# with sequence length of 128 for 90% of the steps. Then, we train
# the rest 10% of the steps of sequence of 512 to learn the positional embeddings."
MAX_LEN = 512
SEQ_LEN = 128
### BookCorpus
VOCAB_PATH = Path(__file__).parent/"pretrain/bookcorpus_vocab.json"
# VOCAB_DIR = Path(__file__).parent/"pretrain/bookcorpus_vocab"
MIN_FREQ = 5
LIM_ALPHABET = 100

### Architecture
DROP_PROB = 0.1 # "We use a dropout probability of 0.1 on all layers."
N_LAYERS = 6
N_HEADS = 6
HIDDEN_SIZE = 384
MLP_SIZE = 384 * 4

### Optimizer
MAX_LR = 1e-4
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.01
N_WARMUP_STEPS = 10_000
# "We use Adam with learning rate of 1e-4, $beta_{1} = 0.9$, $beta_{2} = 0.999$,
# L2 weight decay of 0.01, learning rate warmup over the first 10,000 steps,
# and linear decay of the learning rate."

### Training
N_GPUS = torch.cuda.device_count()
if N_GPUS > 0:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
N_WORKERS = 4
CKPT_DIR = Path(__file__).parent/"checkpoints"
N_CKPT_SAMPLES = 400_000
### Masked Language Model
SELECT_PROB = 0.15
MASK_PROB = 0.8
RANDOMIZE_PROB = 0.1

### Resume
CKPT_PATH = CKPT_DIR/"bookcorpus_step_step_42174.pth"
