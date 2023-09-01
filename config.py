# import sys

# sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/bert_from_scratch")

import torch
from pathlib import Path

### Data
VOCAB_SIZE = 30_522
MAX_LEN = 512
### BookCorpus
VOCAB_PATH = Path(__file__).parent/"pretrain/bookcorpus_vocab.json"
MIN_FREQ = 5

### Architecture
DROP_PROB = 0.1 # "For the base model, we use a rate of $P_{drop} = 0.1$."

### Optimizer
MAX_LR = 1e-4
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.01
N_WARMUP_STEPS = 10_000
# "We use Adam with learning rate of 1e-4,  1 = 0:9, 2 = 0:999, L2 weight decay of 0:01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate. We use a dropout probability of 0.1 on all layers. We use a gelu activation (Hendrycks and Gimpel, 2016) rather than the standard relu, following OpenAI GPT. The training loss is the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood. Training of BERTBASE was performed"

### Training
N_GPUS = torch.cuda.device_count()
if N_GPUS > 0:
    DEVICE = torch.device("cuda")
    print(f"""Using {N_GPUS} GPU(s).""")
else:
    DEVICE = torch.device("cpu")
    print(f"""Using CPU(s).""")
AUTOCAST = False
N_WORKERS = 4
CKPT_DIR = Path(__file__).parent/"checkpoints"
### Masked Language Model
N_CKPT_SAMPLES = 20_000
SELECT_PROB=0.15
MASK_PROB=0.8
RANDOMIZE_PROB=0.1
