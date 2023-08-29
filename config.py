### Data
VOCAB_SIZE = 30_522
MAX_LEN = 512
### BookCorpus
EPUBTXT_DIR = "/Users/jongbeomkim/Documents/datasets/bookcorpus/epubtxt"
# EPUBTXT_DIR = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"
VOCAB_PATH = "/Users/jongbeomkim/Desktop/workspace/bert_from_scratch/pretrain/bookcorpus_vocab.json"

### Architecture
DROP_PROB = 0.1 # "For the base model, we use a rate of $P_{drop} = 0.1$."
# D_MODEL = 512
# N_HEADS = 8
# N_LAYERS = 6
# HIDDEN_SIZE = 768

### Training
BATCH_SIZE = 8
