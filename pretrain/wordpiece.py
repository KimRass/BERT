# References
    # https://huggingface.co/docs/tokenizers/pipeline

from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders
from pathlib import Path

import config


def train_bert_tokenizer(
    vocab_size, vocab_path, min_freq, corpus_files, post_processor=False
):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    if post_processor:
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )
    tokenizer.decoder = decoders.WordPiece()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        limit_alphabet=vocab_size // 5,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    corpus_files = list(map(str, corpus_files))
    tokenizer.train(files=corpus_files, trainer=trainer)
    tokenizer.save(str(vocab_path))


def load_bert_tokenizer(vocab_path):
    tokenizer = Tokenizer.from_file(str(vocab_path))
    return tokenizer


if __name__ == "__main__":
    corpus_files = list(Path(config.EPUBTXT_DIR).glob("*.txt"))
    if not Path(config.VOCAB_PATH).exists():
        train_bert_tokenizer(
            vocab_size=config.VOCAB_SIZE,
            vocab_path=config.VOCAB_PATH,
            min_freq=config.MIN_FREQ,
            corpus_files=corpus_files,
            post_processor=True,
        )
