# "For the pre-training corpus we use the BooksCorpus (800M words) (Zhu et al., 2015)
# and English Wikipedia (2,500M words). For Wikipedia we extract only the text passages
# and ignore lists, tables, and headers. It is critical to use a document-level corpus
# rather than a shuffled sentence-level corpus such in order to extract long contiguous sequences."

import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from pathlib import Path
from tqdm.auto import tqdm

from transformer_based_models.tokenizers.wordpiece import encode

np.set_printoptions(edgeitems=20, linewidth=sys.maxsize, suppress=False)
torch.set_printoptions(edgeitems=16, linewidth=sys.maxsize, sci_mode=True)


class WikipediaForBERT(Dataset):
    def __init__(
        self,
        data_dir,
        # tokenizer,
        vocab,
        max_len,
    ):
