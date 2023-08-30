# References
    # https://huggingface.co/docs/tokenizers/pipeline

from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders


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
    tokenizer.save(vocab_path)


def load_bert_tokenizer(vocab_path):
    tokenizer = Tokenizer.from_file(vocab_path)
    return tokenizer
