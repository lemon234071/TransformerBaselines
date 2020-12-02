import os
import sys
import time
import math
import tqdm
import logging

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, BertConfig, BertForTokenClassification

from utils import Statistics
from agents.optim_schedule import ScheduledOptim
from .data_utils import build_dataset, collate

# BERT_MODEL = 'bert-base-uncased'
BERT_MODEL = 'bert-base-chinese'

logger = logging.getLogger(__file__)


class Trainer(object):

    @classmethod
    def add_cmdline_args(cls, argparser):
        # super(SoftMaskedBertTrainer, cls).add_cmdline_args(argparser)
        ScheduledOptim.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('SoftMaskedBertTrainer Arguments')
        # add_common_cmdline_args(agent)
        # memory and knowledge arguments
        agent.add_argument('--batch_size', default=8, type=int)
        agent.add_argument('--num_workers', default=8, type=int)
        agent.add_argument('--max_len', default=128, type=int)

        agent.add_argument('--vocab_path', type=str, default=None)
        agent.add_argument('--checkpoint', type=str, default=BERT_MODEL)
        agent.add_argument("--hidden_size", default=256, type=int)
        agent.add_argument("--rnn_layer", default=1, type=int)

        agent.add_argument("--learning_rate", default=2e-5, type=float)
        agent.add_argument("--gradient_accumulation_steps", type=int, default=1,
                           help="Accumulate gradients on several steps")
        agent.add_argument("--max_grad_norm", type=float, default=1.0,
                           help="Clipping gradient norm")

        agent.add_argument('--report_every', default=-1, type=int)

        agent.add_argument('--gama', type=float, default=0.8)

    def __init__(self, opt, device):

        self.opt = opt
        self.device = device

        self._dataset = {}
        self._dataloader = {}

        self.tokenizer = BertTokenizer.from_pretrained(opt.vocab_path if opt.vocab_path else opt.checkpoint,
                                                       do_lower_case=True)
        self.config = BertConfig.from_pretrained(opt.checkpoint)
        self.model = BertForTokenClassification.from_pretrained(opt.checkpoint, config=self.config).to(device)

        # if torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for train" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=[0,1,2])

        # _optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        _optimizer = _get_optimizer(self.model, opt)
        self.optim_schedule = ScheduledOptim(opt, _optimizer)

    def load_data(self, datasets, infer=False):
        for k, v in datasets.items():
            # self._dataset[type] = BertDataset(self.tokenizer, data, max_len=self.opt.max_len)
            self._dataset[k] = build_dataset(v, self.tokenizer)
            tensor_dataset = collate(self._dataset[k], self.tokenizer.pad_token_id)
            dataset = TensorDataset(*tensor_dataset)
            shuffle = (k == "train") and not infer
            self._dataloader[k] = DataLoader(dataset,
                                             batch_size=self.opt.batch_size,
                                             num_workers=self.opt.num_workers,
                                             shuffle=shuffle)

    def train(self, epoch, data_type="train"):
        self.model.train()
        return self.iteration(epoch, self._dataloader[data_type])

    def evaluate(self, epoch, data_type="valid"):
        self.model.eval()
        return self.iteration(epoch, self._dataloader[data_type], data_type=data_type)

    def infer(self, data_type):
        data_loader = self._dataloader[data_type]
        self.model.eval()
        out_put = []
        data_loader = tqdm.tqdm(enumerate(data_loader),
                                desc="%s" % 'Inference:',
                                total=len(data_loader),
                                bar_format="{l_bar}{r_bar}")
        stats = Statistics()
        for step, batch in data_loader:
            # 0. batch_data will be sent into the device(GPU or cpu)
            input_ids, input_mask, output_idsz, labels = tuple(
                input_tensor.to(self.device) for input_tensor in batch)

            loss, logits = self.model(input_ids, input_mask, labels=labels)  # prob [batch_size, seq_len, 1]

            self._stats(stats, loss.item(), logits.softmax(dim=-1), labels)

            label_mask = logits.softmax(dim=-1).argmax(dim=-1).bool()
            input_ids[label_mask] = self.tokenizer.mask_token_id
            out_put.extend(
                [line[line_mask.bool()].cpu().tolist()[1:-1] for line, line_mask in zip(input_ids, input_mask)])
        self._report(stats)
        return [''.join(self.tokenizer.convert_ids_to_tokens(x)) for x in out_put]

    def save(self, file_path):
        torch.save(self.model.cpu(), file_path)
        self.model.to(self.device)
        logger.info('Model save {}'.format(file_path))

    def load(self, file_path):
        if not os.path.exists(file_path):
            return
        self.model = torch.load(file_path)
        self.model.to(self.device)

    def iteration(self, epoch, data_loader, data_type="train"):
        str_code = data_type

        # Setting the tqdm progress bar
        data_loader = tqdm.tqdm(enumerate(data_loader),
                                desc="Epoch_%s:%d" % (str_code, epoch),
                                total=len(data_loader),
                                bar_format="{l_bar}{r_bar}")

        logger.info("***** Running *****")
        # logger.info("  Num examples = %d", len(self.train_dataset))
        # logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        # logger.info("  Total train batch size = %d", self.args.train_batch_size)
        # logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        # logger.info("  Total optimization steps = %d", t_total)
        # logger.info("  Logging steps = %d", self.args.logging_steps)
        # logger.info("  Save steps = %d", self.args.save_steps)

        stats = Statistics()
        for step, batch in data_loader:
            # 0. batch_data will be sent into the device(GPU or cpu)
            # data = {key: value.to(self.device) for key, value in data.items()}
            input_ids, input_mask, output_ids, labels = tuple(
                input_tensor.to(self.device) for input_tensor in batch)

            loss, logits = self.model(input_ids, input_mask, labels=labels)  # prob [batch_size, seq_len, 1]

            if data_type == "train":
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward(retain_graph=True)
                if step % self.opt.gradient_accumulation_steps == 0:
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)
                    self.optim_schedule.step()
                    self.optim_schedule.zero_grad()

            # sta
            self._stats(stats, loss.item(), logits.softmax(dim=-1), labels)
            # if data_type == "train" and self.opt.report_every > 0 and step % self.opt.report_every == 0:
            #     post_fix = {
            #         "epoch": epoch,
            #         "iter": step,
            #         "lr": self.optim_schedule.learning_rate()
            #     }
            #     post_fix.update(stats.report())
            #     data_loader.write(
            #         "\n" + str({k: (round(v, 5) if isinstance(v, float) else v) for k, v in post_fix.items()}))
            #     sys.stdout.flush()

        logger.info("Epoch{}_{}, ".format(epoch, str_code))
        self._report(stats)
        return stats.xent()

    def _stats(self, stats: Statistics, loss, d_scores, target):
        d_pred = d_scores.argmax(dim=-1)
        non_padding = target.ne(-100)
        num_non_padding = non_padding.sum().item()

        metrics = {
            # "n_correct": d_pred.eq(target).masked_select(non_padding).sum().item(),
            "d_tp": (d_pred.eq(1) & target.eq(1)).masked_select(non_padding).sum().item(),
            "d_fp": (d_pred.eq(1) & target.eq(0)).masked_select(non_padding).sum().item(),
            "d_tn": (d_pred.eq(0) & target.eq(0)).masked_select(non_padding).sum().item(),
            "d_fn": (d_pred.eq(0) & target.eq(1)).masked_select(non_padding).sum().item(),
        }
        stats.update(loss * num_non_padding, num_non_padding, metrics)
        # stats.update(loss * num_non_padding, 0, d_num_correct, num_non_padding,
        #              tp, fp, tn, fn)

    def _report(self, stats: Statistics):
        logger.info("avg_loss: {} ".format(round(stats.xent(), 5)) +
                    "acc: {}, prec: {}, recall: {}, f1: {}".format(
                        round(stats.aprf("d_")[0], 5), round(stats.aprf("d_")[1], 5),
                        round(stats.aprf("d_")[2], 5), round(stats.aprf("d_")[3], 5))
                    )


def _get_optimizer(model, opt):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': opt.weight_decay},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optim.Adam(optimizer_grouped_parameters, lr=opt.learning_rate)
