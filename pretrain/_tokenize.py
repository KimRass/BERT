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
    vocab_size, vocab_path, corpus_files, post_processor=False
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
        vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    corpus_files = list(map(str, corpus_files))
    tokenizer.train(files=corpus_files, trainer=trainer)
    tokenizer.save(vocab_path)


def load_bert_tokenizer(vocab_path):
    tokenizer = Tokenizer.from_file(vocab_path)
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
