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

VOCAB_SIZE = 30_522


def prepare_bert_tokenizer(vocab_path, corpus_files=None, post_processor=False):
    if not Path(vocab_path).exists():
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordPieceTrainer(
            vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        tokenizer.train(files=corpus_files, trainer=trainer)
        tokenizer.save(vocab_path)

    tokenizer = Tokenizer.from_file(vocab_path)
    tokenizer.decoder = decoders.WordPiece()

    if post_processor:
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )
    return tokenizer


if __name__ == "__main__":
    corpus_files = [
        f"""/Users/jongbeomkim/Documents/datasets/wikitext-103-raw/wiki.{split}.raw"""
        # for split in ["test", "train", "valid"]
        for split in ["train"]
    ]
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path, corpus_files=corpus_files)

    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(output.tokens, output.ids)
