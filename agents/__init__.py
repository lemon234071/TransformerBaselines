from .bert_agents import *

AGENT_CLASSES = {
    'BERT_c': bert_c_trainer,
    'SoftMaskedBERT': soft_masked_bert_trainer,
    "BERT_sl": sequence_labeling_trainer
}

DATA_CLASSES = {
    'BERT_c': bert_c_dataloader,
    'SoftMaskedBERT': soft_masked_bert_dataloader,
    "BERT_sl": sequence_labeling_dataloader
}
