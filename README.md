# Paper Reading
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)

# Research
- BERT 논문의 내용처럼 `Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)`로 할 경우 Pre-training 진행되지 않는 현상을 발견했습니다. 아무리 학습을 시켜도 NSP에 있어서 Loss 값이 0.69 이하로 떨어지지 않았습니다.
