#!usr/bin/env python
# -*- coding:utf-8 -*-
import os
import tqdm
import random
import logging
import argparse
from pprint import pformat

import numpy as np
import pandas as pd
import torch

from agents.bert_agents.soft_masked_bert.trainer import SoftMaskedBertTrainer
from agents.bert_agents.soft_masked_bert.data_process import get_datasets

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # - %(name)s
logger = logging.getLogger(__file__)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
logger.info("Device: {}".format(device))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)

parser = argparse.ArgumentParser()
# data
parser.add_argument("--dataset_path", type=str, default="data/xiaowei/neg/",
                    help="Path or url of the dataset. If empty download accroding to dataset.")
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset name")
parser.add_argument("--save_dir", type=str, default="checkpoints")

# training
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--early_stop', default=10, type=int)


def main():
    # my_module = importlib.import_module(module_name)
    # model_class = getattr(my_module, class_name)
    SoftMaskedBertTrainer.add_cmdline_args(parser)
    opt = parser.parse_args()
    logger.info("Arguments: %s", pformat(opt))

    trainer = SoftMaskedBertTrainer(opt, device)

    datasets = get_datasets(opt.dataset_path)
    trainer.load_data(datasets)

    best_checkpoint = SoftMaskedBertTrainer.__name__ + opt.dataset + 'best_model.pt'
    best_loss = 10000
    patience = 0
    for e in range(opt.epochs):
        trainer.train(e)
        val_loss = trainer.evaluate(e, "valid")
        if best_loss > val_loss:
            best_loss = val_loss
            trainer.save(best_checkpoint)
            logger.info('Best val loss {} at epoch {}'.format(best_loss, e))
            test_loss = trainer.evaluate(e, "test")
        else:
            patience += 1
            if patience > opt.early_stop:
                break

        trainer.load(best_checkpoint)
        # for i in trainer.inference(val):
        #     print(i)
        #     print('\n')
        # if do_test:
        #     final_test_ppl = manager.evaluate('test')
        #     print('Test PPL: {:.4f}'.format(final_test_ppl))


if __name__ == '__main__':
    main()
