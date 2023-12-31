# References
    # https://huggingface.co/docs/tokenizers/pipeline

from transformers import BertTokenizerFast
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders
from pathlib import Path
from tqdm.auto import tqdm
import re
import argparse

import config
from utils import REGEX


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epubtxt_dir", type=str, required=False, default="../bookcurpus/epubtxt",
    )

    args = parser.parse_args()
    return args


def parse(epubtxt_dir, with_document=False):
    print("Parsing BookCorpus...")
    lines = list()
    for doc_path in tqdm(list(Path(epubtxt_dir).glob("*.txt"))):
        with open(doc_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if (not line) or (re.search(pattern=REGEX, string=line)) or (line.count(" ") < 1):
                    continue
                if not with_document:
                    lines.append(line)
                else:
                    lines.append((doc_path.name, line))
    print("Completed.")
    print(f"Number of paragraphs: {len(lines):,}")
    return lines


def train_bert_tokenizer(corpus, vocab_size, vocab_path, min_freq, limit_alphabet):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.WordPiece()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        limit_alphabet=limit_alphabet,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.train_from_iterator(iterator=corpus, trainer=trainer)
    tokenizer.save(str(vocab_path))


def load_bert_tokenizer(vocab_path):
    tokenizer = Tokenizer.from_file(str(vocab_path))
    return tokenizer


def train_fast_bert_tokenizer(corpus, vocab_size, vocab_dir):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer = tokenizer.train_new_from_iterator(corpus, vocab_size=vocab_size, length=len(corpus))
    tokenizer.save_pretrained(vocab_dir)


def load_fast_bert_tokenizer(vocab_dir):
    tokenizer = BertTokenizerFast.from_pretrained(vocab_dir)
    return tokenizer


if __name__ == "__main__":
    args = get_args()

    corpus = parse(args.epubtxt_dir)
    train_fast_bert_tokenizer(
        corpus=corpus,
        vocab_size=config.VOCAB_SIZE,
        vocab_dir=config.VOCAB_DIR,
    )
    tokenizer = load_fast_bert_tokenizer(vocab_dir=config.VOCAB_DIR)
    # if not Path(config.VOCAB_PATH).exists():
    #     train_bert_tokenizer(
    #         corpus=corpus,
    #         vocab_size=config.VOCAB_SIZE,
    #         vocab_path=config.VOCAB_PATH,
    #         min_freq=config.MIN_FREQ,
    #         limit_alphabet=config.LIM_ALPHABET,
    #         post_processor=False,
    #     )
