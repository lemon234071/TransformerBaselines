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
from transformers import BertTokenizer, BertModel, BertConfig

from utils import Statistics
from agents.optim_schedule import ScheduledOptim
from .soft_masked_bert import SoftMaskedBert
from .data_utils import build_dataset, collate

# BERT_MODEL = 'bert-base-uncased'
BERT_MODEL = 'bert-base-chinese'

logger = logging.getLogger(__file__)


class SoftMaskedBertTrainer(object):

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
        self.model = SoftMaskedBert(opt, self.tokenizer, device).to(device)

        # if torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for train" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=[0,1,2])

        # _optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        _optimizer = _get_optimizer(self.model, opt)
        self.optim_schedule = ScheduledOptim(opt, _optimizer)

        self.criterion_c = nn.NLLLoss(ignore_index=self.tokenizer.pad_token_id)
        self.criterion_d = nn.BCELoss(reduction="none")
        self.gama = opt.gama

    def load_data(self, datasets):
        for k, v in datasets.items():
            # self._dataset[type] = BertDataset(self.tokenizer, data, max_len=self.opt.max_len)
            self._dataset[k] = build_dataset(v, self.tokenizer)
            tensor_dataset = collate(self._dataset[k], self.tokenizer.pad_token_id)
            dataset = TensorDataset(*tensor_dataset)
            self._dataloader[k] = DataLoader(dataset,
                                             batch_size=self.opt.batch_size,
                                             num_workers=self.opt.num_workers,
                                             shuffle=(k == "train"))

    def train(self, epoch, data_type="train"):
        self.model.train()
        return self.iteration(epoch, self._dataloader[data_type])

    def evaluate(self, epoch, data_type="valid"):
        self.model.eval()
        return self.iteration(epoch, self._dataloader[data_type], data_type=data_type)

    def infer(self, data_loader):
        self.model.eval()
        out_put = []
        data_loader = tqdm.tqdm(enumerate(data_loader),
                                desc="%s" % 'Inference:',
                                total=len(data_loader),
                                bar_format="{l_bar}{r_bar}")
        for i, data in data_loader:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            out, prob = self.model(data["input_ids"], data["input_mask"],
                                   data["segment_ids"])  # prob [batch_size, seq_len, 1]
            out_put.extend(out.argmax(dim=-1).cpu().numpy().tolist())
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

        stats = Statistics()
        for step, batch in data_loader:
            # 0. batch_data will be sent into the device(GPU or cpu)
            # data = {key: value.to(self.device) for key, value in data.items()}
            input_ids, input_mask, output_ids, labels = tuple(
                input_tensor.to(self.device) for input_tensor in batch)

            d_scores, c_scores = self.model(input_ids, input_mask)  # prob [batch_size, seq_len, 1]
            d_scores = d_scores.squeeze(dim=-1)
            loss_d = self.criterion_d(d_scores, labels.float())
            label_mask = labels.ne(-1)
            loss_d = (loss_d * label_mask).sum() / label_mask.float().sum()
            loss_c = self.criterion_c(c_scores.view(-1, c_scores.size(-1)), output_ids.view(-1))
            loss = (1 - self.gama) * loss_d + self.gama * loss_c

            if data_type == "train":
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward(retain_graph=True)
                if step % self.opt.gradient_accumulation_steps == 0:
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)
                    self.optim_schedule.step()
                    self.optim_schedule.zero_grad()

            # sta
            self._stats(stats, loss.item(), d_scores, labels, c_scores, input_ids, output_ids)
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

    def _stats(self, stats, loss, d_scores, label, c_scores, inputs, target):
        c_pred = c_scores.argmax(dim=-1)
        non_padding = target.ne(self.tokenizer.pad_token_id)
        num_non_padding = non_padding.sum().item()
        d_pred = torch.round(d_scores).long()

        error = target.ne(inputs)

        metrics = {
            "d_tp": (d_pred.eq(1) & error.eq(True)).masked_select(non_padding).sum().item(),
            "d_tn": (d_pred.eq(0) & error.eq(False)).masked_select(non_padding).sum().item(),
            "d_fp": (d_pred.eq(1) & error.eq(False)).masked_select(non_padding).sum().item(),
            "d_fn": (d_pred.eq(0) & error.eq(True)).masked_select(non_padding).sum().item(),
            # "n_correct": c_pred.eq(target).masked_select(non_padding).sum().item(),
            "c_tp": (c_pred.eq(target) & error.eq(True)).masked_select(non_padding).sum().item(),
            "c_tn": (c_pred.eq(target) & error.eq(False)).masked_select(non_padding).sum().item(),
            "c_fp": (c_pred.ne(target) & error.eq(False)).masked_select(non_padding).sum().item(),
            "c_fn": (c_pred.ne(target) & error.eq(True)).masked_select(non_padding).sum().item(),
        }
        stats.update(loss * num_non_padding, num_non_padding, metrics)

    def _report(self, stats: Statistics):
        logger.info("avg_loss: {} ".format(round(stats.xent(), 5)) +
                    "acc: {}, prec: {}, recall: {}, f1: {}".format(
                        round(stats.aprf("d_")[0], 5), round(stats.aprf("d_")[1], 5),
                        round(stats.aprf("d_")[2], 5), round(stats.aprf("d_")[3], 5)) +
                    "acc: {}, prec: {}, recall: {}, f1: {}".format(
                        round(stats.aprf("c_")[0], 5), round(stats.aprf("c_")[1], 5),
                        round(stats.aprf("c_")[2], 5), round(stats.aprf("c_")[3], 5))
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
