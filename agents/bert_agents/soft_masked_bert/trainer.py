import os
import tqdm
import logging

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, BertConfig

from agents.optim_schedule import ScheduledOptim
from .soft_masked_bert import SoftMaskedBert
from .data_utils import build_dataset, collate

BERT_MODEL = 'bert-base-uncased'

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
        agent.add_argument("--max_norm", type=float, default=1.0,
                           help="Clipping gradient norm")

        agent.add_argument('--report_every', default=100, type=int)

        agent.add_argument('--gama', type=float, default=0.8)

    def __init__(self, opt, device):

        self.opt = opt
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(opt.vocab_path if opt.vocab_path else opt.checkpoint,
                                                       do_lower_case=True)
        self.model = SoftMaskedBert(opt, self.tokenizer, device).to(device)

        # if torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for train" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=[0,1,2])

        # _optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        _optimizer = _get_optimizer(self.model, opt.learning_rate)
        self.optim_schedule = ScheduledOptim(opt, _optimizer)

        self.criterion_c = nn.NLLLoss()
        self.criterion_d = nn.BCELoss()
        self.gama = opt.gama
        self.log_freq = opt.report_every

        self._dataset = {}
        self._dataloader = {}

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

    def train(self, epoch):
        self.model.train()
        return self.iteration(epoch, self._dataloader["train"])

    def evaluate(self, epoch):
        self.model.eval()
        return self.iteration(epoch, self._dataloader["valid"], train=False)

    def inference(self, data_loader):
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
        print('Model save {}'.format(file_path))

    def load(self, file_path):
        if not os.path.exists(file_path):
            return
        self.model = torch.load(file_path)
        self.model.to(self.device)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "val"

        # Setting the tqdm progress bar
        data_loader = tqdm.tqdm(enumerate(data_loader),
                                desc="EP_%s:%d" % (str_code, epoch),
                                total=len(data_loader),
                                bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        # total_correct = 0
        total_element = 0
        c_correct = 0
        d_correct = 0

        for i, batch in data_loader:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}
            input_ids, input_mask, token_type_ids, labels = tuple(
                input_tensor.to(self.device) for input_tensor in batch)

            out, prob = self.model(data["input_ids"], data["input_mask"],
                                   data["segment_ids"])  # prob [batch_size, seq_len, 1]
            prob = prob.reshape(-1, prob.shape[1])
            loss_d = self.criterion_d(prob, data['label'].float())
            loss_c = self.criterion_c(out.transpose(1, 2), data["output_ids"])
            loss = self.gama * loss_c + (1 - self.gama) * loss_d

            if train:
                self.optim_schedule.zero_grad()
                loss.backward(retain_graph=True)
                self.optim_schedule.step()

            # correct = out.argmax(dim=-1).eq(data["output_ids"]).sum().item()
            out = out.argmax(dim=-1)
            c_correct += sum([out[i].equal(data['output_ids'][i]) for i in range(len(out))])
            prob = torch.round(prob).long()
            d_correct += sum([prob[i].equal(data['label'][i]) for i in range(len(prob))])

            avg_loss += loss.item()
            #     total_correct += c_correct
            #     # total_element += data["label"].nelement()
            total_element += len(data)

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "d_acc": d_correct / total_element,
                "c_acc": c_correct / total_element
            }

            if i % self.log_freq == 0:
                data_loader.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_loader), "d_acc=",
              d_correct / total_element, "c_acc", c_correct / total_element)
        return avg_loss / len(data_loader)


def _get_optimizer(model, learning_rate):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optim.Adam(optimizer_grouped_parameters, lr=learning_rate)
