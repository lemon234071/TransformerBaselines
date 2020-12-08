import os
import tqdm
import logging
import torch

from utils import Statistics
from agents.optim_schedule import ScheduledOptim

logger = logging.getLogger(__file__)


class BaseTrainer(object):

    @classmethod
    def add_cmdline_args(cls, argparser):
        ScheduledOptim.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('BaseTrainer Arguments')
        # add_common_cmdline_args(agent)
        # memory and knowledge arguments
        agent.add_argument('--batch_size', default=8, type=int)
        agent.add_argument('--num_workers', default=8, type=int)
        agent.add_argument('--max_len', default=128, type=int)

        agent.add_argument('--vocab_path', type=str, default=None)
        agent.add_argument("--hidden_size", default=256, type=int)

        agent.add_argument("--learning_rate", default=2e-5, type=float)
        agent.add_argument("--gradient_accumulation_steps", type=int, default=1,
                           help="Accumulate gradients on several steps")
        agent.add_argument("--max_grad_norm", type=float, default=1.0,
                           help="Clipping gradient norm")
        agent.add_argument('--skip_report_eval_steps', default=0, type=int)

        agent.add_argument('--report_every', default=-1, type=int)

    def __init__(self, opt, device):

        self.opt = opt
        self.device = device

        self._dataset = {}
        self._dataloader = {}

    def load_data(self, datasets, infer=False):
        raise NotImplementedError

    def train(self, epoch, data_type="train"):
        self.model.train()
        return self.iteration(epoch, self._dataloader[data_type])

    def evaluate(self, epoch, data_type="valid"):
        self.model.eval()
        return self.iteration(epoch, self._dataloader[data_type], data_type=data_type)

    def infer(self, data_type):
        raise NotImplementedError

    def save(self, file_path):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(file_path)
        self.tokenizer.save_pretrained(file_path)

        # Save training arguments together with the trained model
        torch.save(self.opt, os.path.join(file_path, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", file_path)

    def load(self, file_path):
        if not os.path.exists(file_path):
            raise Exception("%s does not exists" % file_path)
        try:
            self.model = self.model.from_pretrained(file_path)
            self.model.to(self.device)
            logger.info("***** Model Loaded from {} *****".format(file_path))
        except:
            raise Exception("Some model files might be missing...")

    def iteration(self, epoch, data_loader, data_type="train"):
        raise NotImplementedError
