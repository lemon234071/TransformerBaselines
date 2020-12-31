import os
import logging
import torch

from agents.optim_schedule import ScheduledOptim, _get_optimizer

logger = logging.getLogger(__file__)


class BaseTrainer(object):

    @classmethod
    def add_cmdline_args(cls, argparser):
        ScheduledOptim.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('BaseTrainer Arguments')
        # add_common_cmdline_args(agent)
        # memory and knowledge arguments
        agent.add_argument('--dataset_cache', action='store_true')
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
        agent.add_argument('--skip_report_eval_steps', default=-1, type=int)
        agent.add_argument('--eval_every', default=-1, type=int)
        agent.add_argument('--report_every', default=-1, type=int)

    def __init__(self, opt, device):

        self.opt = opt
        self.device = device

        self.dataset = {}
        self._dataloader = {}

        self.gradient_accumulation_steps = opt.gradient_accumulation_steps
        self.performance = {}
        self.best_performance = -float("inf")
        self.last_performance = -float("inf")
        self.test_performance = -float("inf")
        self.patience = 0
        self.lr_reduce_patience = opt.lr_reduce_patience
        self.lr_decay = opt.lr_decay
        self.best_checkpoint_path = opt.best_checkpoint_path

    def load_data(self, data_type, dataset, build_dataset, infer=False):
        raise NotImplementedError

    def set_optim_schedule(self):
        self._optimizer = _get_optimizer(self.model, self.opt)
        self.optim_schedule = ScheduledOptim(self.opt, self._optimizer,
                                             self.opt.epochs * len(
                                                 self._dataloader["train"]) / self.gradient_accumulation_steps)

    def train_epoch(self, epoch, data_type="train"):
        self.model.train()
        return self.iteration(epoch, self._dataloader[data_type])

    def evaluate(self, epoch, data_type="valid"):
        self.model.eval()
        valid_performance = self.iteration(epoch, self._dataloader[data_type], data_type=data_type)
        if data_type == "valid":
            if valid_performance > self.best_performance:
                self.best_performance = valid_performance
                self.save(self.best_checkpoint_path)
                logger.info(
                    'Best valid performance {} at epoch {}'.format(abs(self.best_performance), epoch))

                if data_type == "valid" and "test" in self._dataloader:
                    test_performance = self.iteration(epoch, self._dataloader["test"], "test")
                    logger.info(
                        'Test performance {} at epoch {}'.format(test_performance, epoch))
                    self.test_performance = test_performance
                self.patience = 0
            elif valid_performance > self.last_performance:
                logger.info('Valid performance {} better than last'.format(abs(valid_performance)))
            else:
                self.patience += 1
                if self.patience >= self.lr_reduce_patience > 0:
                    lr = self.optim_schedule.get_lr()
                    self.optim_schedule.set_lr(lr * self.lr_decay)
                    logger.info("lr decayed from {} to {} with patinece {}".format(lr, self.optim_schedule.get_lr(),
                                                                                   self.patience))
            self.last_performance = valid_performance

        return valid_performance

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
