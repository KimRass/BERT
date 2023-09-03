import torch
import torch.nn as nn


class QuestionAnsweringHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        # "We only introduce a start vector $S \in \mathbb{R}^{H}$ and an end vector
        # $E \in \mathbb{R}^{H}$ during fine-tuning."
        self.proj = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # "The probability of word $i$ being the start of the answer span is computed
        # as a dot product between $T_{i}$ and $S$ followed by a softmax over all of the words
        # in the paragraph."
        x = self.proj(x)
        start_logit, end_logit = torch.split(x, split_size_or_sections=1, dim=2)
        start_logit, end_logit = start_logit.squeeze(), end_logit.squeeze()
        start_id, end_id = torch.argmax(start_logit, dim=1), torch.argmax(end_logit, dim=1)
        return start_id, end_id


class MultipleChoiceHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.proj(x)
        x = x.squeeze()
        x = torch.argmax(x, dim=1)
        return x