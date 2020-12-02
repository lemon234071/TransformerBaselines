#!usr/bin/env python
# -*- coding:utf-8 -*-
import os
import random
import logging
import argparse
import importlib
import platform
from pprint import pformat

import numpy as np
import torch

from utils import *

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # - %(name)s
logger = logging.getLogger(__file__)

device = torch.device('cuda' if torch.cuda.is_available() and platform.system() != 'Windows' else 'cpu')
logger.info("Device: {}".format(device))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)

parser = argparse.ArgumentParser()
# agent
parser.add_argument("--agent", type=str, required=True,
                    help="Agent name")

# data
parser.add_argument("--dataset_path", type=str, default="data/catslu/map/",
                    help="Path or url of the dataset. If empty download accroding to dataset.")
parser.add_argument("--save_dir", type=str, default="checkpoint/")
parser.add_argument('--save_name', type=str, default="")

# training
parser.add_argument('--epochs', default=100000, type=int)
parser.add_argument('--early_stop', default=3, type=int)
parser.add_argument('--mode', type=str, default="train")

# model
parser.add_argument('--result_path', type=str, default="")


def get_agent(agent_name):
    # "agents.bert_agents.sequence_labeling"
    trainer_module = importlib.import_module("agents." + agent_name + ".trainer")
    trainer_class = getattr(trainer_module, "Trainer")
    getdata_module = importlib.import_module("agents." + agent_name + ".data_process")
    getdata_class = getattr(getdata_module, "get_datasets")
    return trainer_class, getdata_class


parsed = vars(parser.parse_known_args()[0])
# trainer_class, getdata_class = AGENT_CLASSES[parsed.get('agent')]
trainer_class, getdata_class = get_agent(parsed.get('agent'))
trainer_class.add_cmdline_args(parser)
opt = parser.parse_args()


def main():
    # my_module = importlib.import_module(module_name)
    # model_class = getattr(my_module, class_name)

    logger.info("Arguments: %s", pformat(opt))
    trainer = trainer_class(opt, device)
    datasets = getdata_class(opt.dataset_path)
    trainer.load_data(datasets)

    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    best_checkpoint = opt.save_dir + opt.save_name + "&&&" + parsed.get('agent') + "_" + \
                      opt.dataset_path.replace("/", "&&&").replace("\\", "&&&") + '_best_model.pt'

    if opt.mode == "infer":
        if not os.path.exists(opt.checkpoint):
            opt.checkpoint = best_checkpoint
        trainer.load(opt.checkpoint)
        result = trainer.infer("test")
        if opt.result_path:
            save_json(result, opt.result_path)
    else:
        best_loss = -10000
        patience = 0
        for e in range(opt.epochs):
            trainer.train(e)
            trainer.show_case = True
            val_loss = trainer.evaluate(e, "valid")
            if best_loss < val_loss:
                best_loss = val_loss
                trainer.save(best_checkpoint)
                logger.info('Best val loss {} at epoch {}'.format(best_loss, e))
                test_loss = trainer.evaluate(e, "test")
                patience = 0
            else:
                patience += 1
                if patience >= opt.early_stop // 2:
                    trainer.optim_schedule.set_lr(trainer.optim_schedule.get_lr() * 0.5)
                    print(trainer.optim_schedule.get_lr(), patience, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if patience > opt.early_stop:
                    break


if __name__ == '__main__':
    main()
