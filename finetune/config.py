# import sys

# sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/bert_from_scratch")

import torch
from pathlib import Path

### Data
VOCAB_SIZE = 30_522 // 2
VOCAB_PATH = Path(__file__).parent.parent/"pretrain/bookcorpus_vocab.json"
MAX_LEN = 512
SEQ_LEN = 128
N_CHOICES = 4

### Architecture
DROP_PROB = 0.1 # "We use a dropout probability of 0.1 on all layers."
N_LAYERS = 6
N_HEADS = 6
HIDDEN_SIZE = 384
MLP_SIZE = 384 * 4

### Optimizer
N_EPOCHS = 3
LR = 2e-5
BETA1 = 0.9
BETA2 = 0.999

### Training
N_GPUS = torch.cuda.device_count()
if N_GPUS > 0:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
N_WORKERS = 0
CKPT_DIR = Path(__file__).parent/"checkpoints"
N_CKPT_SAMPLES = 400_000
