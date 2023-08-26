# Paper Reading
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
## Train
### Pre-training
- We train with batch size of 256 sequences (256 sequences * 512 tokens = 128,000 tokens/batch) for 1,000,000 steps, which is approximately 40 epochs over the 3.3 billion word corpus. (Comment: 256 * 512 * 1,000,000 / 3,300,000,000 = 39.7)
