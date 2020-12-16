# TransformerBaselines

实现基于transformer的baselines
已包括：
1. SoftMaskedBert: Spelling Error Correction with Soft-Masked BERT https://arxiv.org/pdf/2005.07421.pdf
2. T5/MT5
3. CT5 (https://huggingface.co/lemon234071/ct5-small): 阶段版的中文MT5，截断词表和embedding后总计35362词覆盖中文bert词表，自测在效果不差于mt5，截断方式参考的https://github.com/bojone/t5_in_bert4keras。