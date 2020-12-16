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

from agents.utils import *

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
parser.add_argument("--task", type=str, required=True,
                    help="Agent name")

# data
parser.add_argument("--dataset_path", type=str, default="data/catslu/hyps/map/",
                    help="Path or url of the dataset. If empty download accroding to dataset.")
parser.add_argument("--save_dir", type=str, default="checkpoint/")
parser.add_argument('--save_name', type=str, default="")

# training
parser.add_argument('--epochs', required=True)
parser.add_argument('--early_stop', default=-1, type=int)
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--lr_reduce_patience', default=-1, type=int)
parser.add_argument('--lr_decay', type=float, default=0.5)

# infer
parser.add_argument('--result_path', type=str, default="")
parser.add_argument('--infer_data', type=str, default="test")


def get_agent_task(opt):
    agent_name = opt.get('agent')
    task_name = opt.get('task')
    # "agents.bert_agents.sequence_labeling"
    trainer_module = importlib.import_module("agents." + agent_name + ".trainer")
    trainer_class = getattr(trainer_module, "Trainer")
    data_module = importlib.import_module("tasks." + task_name)
    getdata_class = getattr(data_module, "get_datasets")
    builddata_class = getattr(data_module, "build_dataset")
    return trainer_class, getdata_class, builddata_class


parsed = vars(parser.parse_known_args()[0])
# trainer_class, getdata_class = AGENT_CLASSES[parsed.get('agent')]
trainer_class, getdata_class, builddata_class = get_agent_task(parsed)
trainer_class.add_cmdline_args(parser)
opt = parser.parse_args()


def main():
    # my_module = importlib.import_module(module_name)
    # model_class = getattr(my_module, class_name)

    logger.info("Arguments: %s", pformat(opt))

    trainer = trainer_class(opt, device)

    datasets = getdata_class(opt.dataset_path)
    for k, v in datasets.items():
        trainer.load_data(k, v, builddata_class, infer=opt.mode == "infer")
    trainer.set_optim_schedule()

    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    best_checkpoint = opt.save_dir + opt.save_name + "_" + parsed.get('task') + "_" + parsed.get(
        'agent') + '_best_model'
    trainer.bes_checkpoint_path = best_checkpoint

    if opt.mode == "infer":
        if os.path.exists(best_checkpoint):
            opt.checkpoint = best_checkpoint
        logger.info("load checkpoint from {} ".format(opt.checkpoint))
        trainer.load(opt.checkpoint)
        if opt.infer_data not in trainer.dataset:
            raise Exception("%s does not exists in datasets" % opt.infer_data)
        result = trainer.infer(opt.infer_data)
        if opt.result_path:
            save_json(result, opt.result_path)
    else:
        for e in range(opt.epochs):
            trainer.train_epoch(e)
            if trainer.patience >= opt.early_stop > 0:
                break
            trainer.evaluate(e, "valid")
            if trainer.patience >= opt.early_stop > 0:
                break
        logger.info('Test performance {}'.format(trainer.test_performance))


if __name__ == '__main__':
    main()
