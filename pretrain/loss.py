import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader

# from bert.model import BERT, MaskedLanguageModelHead, NextSentencePredictionHead
# from bert.pretrain.bookcorpus import BookCorpusForBERT


class PretrainingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mlm_pred, nsp_pred, token_ids, is_next):
        raw_mlm_loss = F.cross_entropy(
            mlm_pred.permute(0, 2, 1), token_ids, reduction="none",
        )
        # raw_mlm_loss *= gt["prediction_target"]
        mlm_loss = raw_mlm_loss.sum()

        nsp_loss = F.cross_entropy(nsp_pred, is_next)
        loss = mlm_loss + nsp_loss
        return loss


# if __name__ == "__main__":
#     VOCAB_SIZE = 30_522
#     bert = BERT(vocab_size=VOCAB_SIZE)
#     mlm_head = MaskedLanguageModelHead(vocab_size=VOCAB_SIZE)
#     nsp_head = NextSentencePredictionHead()
    
#     criterion = PretrainingLoss()

#     vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab.json"
#     corpus_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"

#     SEQ_LEN = 512
#     # BATCH_SIZE = 256
#     BATCH_SIZE = 8
#     ds = BookCorpusForBERT(vocab_path=vocab_path, corpus_dir=corpus_dir, seq_len=SEQ_LEN)
#     dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#     for batch, data in enumerate(dl, start=1):
#         bert_out = bert(seq=data["masked_ids"], seg_gt=data["segment_gt"])
#         mlm_pred = mlm_head(bert_out)
#         nsp_pred = nsp_head(bert_out)
        
#         criterion(mlm_pred=mlm_pred, nsp_pred=nsp_pred, gt=data)
        
#         bert_out.shape, mlm_pred.shape, nsp_pred.shape
