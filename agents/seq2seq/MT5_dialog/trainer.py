import os
import tqdm
import json
import logging
import platform
import itertools

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Tokenizer, MT5Config, MT5ForConditionalGeneration

from agents.utils import Statistics
from agents.trainer_base import BaseTrainer
from agents.data_utils import collate

logger = logging.getLogger(__file__)


class Trainer(BaseTrainer):

    @classmethod
    def add_cmdline_args(cls, argparser):
        super(Trainer, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('MT5 Arguments')
        # add_common_cmdline_args(agent)
        # memory and knowledge arguments

        agent.add_argument('--checkpoint', type=str, default="google/mt5-small")
        agent.add_argument('--num_beams', type=int, default=1)
        agent.add_argument('--with_label', type=bool, default=False)

    def __init__(self, opt, device):
        super(Trainer, self).__init__(opt, device)
        self.tokenizer = T5Tokenizer.from_pretrained(opt.vocab_path
                                                     if opt.vocab_path else opt.checkpoint, do_lower_case=True)
        self.config = MT5Config.from_pretrained(opt.checkpoint)

        self.model = MT5ForConditionalGeneration(self.config).to(device) \
            if platform.system() == 'Windows' else \
            MT5ForConditionalGeneration.from_pretrained(opt.checkpoint, config=self.config).to(device)
        # raise Exception("handle the embedding")
        # if os.path.isdir(opt.checkpoint):
        #     self.model.
        # if torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for train" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=[0,1,2])

        # _optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        self.skip_report_eval_steps = opt.skip_report_eval_steps
        self.num_beams = opt.num_beams
        self.with_label = opt.with_label

    def truncate_MT5(self, keep_tokens_path):
        keep_tokens = json.load(open(keep_tokens_path))
        print("truncated vocab size: ", len(keep_tokens))
        self.model.config.vocab_size = len(keep_tokens)
        self.model.shared.num_embeddings = len(keep_tokens)
        self.model.shared.weight.data = self.model.shared.weight[keep_tokens].data
        self.model.lm_head.out_features = len(keep_tokens)
        self.model.lm_head.weight.data = self.model.lm_head.weight[keep_tokens].data
        self.save("checkpoint/truncated_MT5/")

    def load_data(self, data_type, dataset, build_dataset, infer=False):
        if self.opt.dataset_cache and data_type == "train":
            dataset_cache = self.opt.save_name + "_" + self.opt.task + '_' + type(self.tokenizer).__name__
            if os.path.isfile(dataset_cache):
                logger.info("Load tokenized train dataset from cache at %s", dataset_cache)
                dataset = torch.load(dataset_cache)
            else:
                dataset = build_dataset(data_type, dataset, self.tokenizer)
                torch.save(dataset, dataset_cache)
        else:
            dataset = build_dataset(data_type, dataset, self.tokenizer)
        self.dataset[data_type] = dataset
        tensor_dataset = collate(self.dataset[data_type], self.tokenizer.pad_token_id, data_type)
        dataset = TensorDataset(*tensor_dataset)
        self._dataloader[data_type] = DataLoader(dataset,
                                                 batch_size=self.opt.batch_size,
                                                 num_workers=self.opt.num_workers,
                                                 shuffle=(data_type == "train" and not infer))

    def infer(self, data_type):
        data_loader = self._dataloader[data_type]
        self.model.eval()
        out_put = []
        data_loader = tqdm.tqdm(enumerate(data_loader),
                                desc="%s" % 'Inference:',
                                total=len(data_loader),
                                bar_format="{l_bar}{r_bar}")
        # stats = Statistics()
        for step, batch in data_loader:
            input_ids, input_mask, labels = tuple(
                input_tensor.to(self.device) for input_tensor in batch)
            generated = self.model.generate(input_ids, attention_mask=input_mask, num_beams=self.num_beams,
                                            num_return_sequences=self.num_beams)
            dec = self.tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if self.with_label:
                da_labels_dec = self.tokenizer.batch_decode(input_ids[:, 3:], skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)
                da_labels_dec = list(itertools.chain(*[[x for _ in range(self.num_beams)] for x in da_labels_dec]))
                dec = [[x, y] for x, y in zip(dec, da_labels_dec)]

            out_put.extend(dec)
            # self._stats(stats, loss.item(), logits.softmax(dim=-1), labels)

        # self._report(stats)
        return out_put

    def iteration(self, epoch, data_loader, data_type="train"):
        str_code = data_type

        # Setting the tqdm progress bar
        data_loader = tqdm.tqdm(enumerate(data_loader),
                                desc="Epoch_%s:%d" % (str_code, epoch),
                                total=len(data_loader),
                                bar_format="{l_bar}{r_bar}",
                                mininterval=2)

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
            input_ids, input_mask, labels = tuple(
                input_tensor.to(self.device) for input_tensor in batch)

            outputs = self.model(input_ids, attention_mask=input_mask, labels=labels,
                                 return_dict=True)  # prob [batch_size, seq_len, 1]
            loss, logits = outputs.loss, outputs.logits

            if data_type == "train":
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()
                if step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)
                    self.optim_schedule.step()
                    self.optim_schedule.zero_grad()
                    if self.opt.eval_every > 0 and step % (
                            self.opt.eval_every * self.opt.gradient_accumulation_steps) == 0:
                        self.evaluate(epoch)
                        if self.patience >= self.opt.early_stop > 0:
                            return
                        self.model.train()
                    data_loader.set_postfix(loss=loss.item(), lr=self.optim_schedule.get_lr(),
                                            step=self.optim_schedule.training_step)

            # sta
            self._stats(stats, loss.item(), logits.softmax(dim=-1).argmax(dim=-1), labels)

        logger.info("Epoch{}_{}, ".format(epoch, str_code))
        self._report(stats, data_type, epoch)

        return -round(stats.xent(), 6)

    def _stats(self, stats: Statistics, loss, preds, target):
        non_padding = target.ne(-100)
        num_non_padding = non_padding.sum().item()
        metrics = {
            "n_correct": preds.eq(target).masked_select(non_padding).sum().item()
        }
        stats.update(loss * num_non_padding, num_non_padding, metrics)

    def _report(self, stats: Statistics, mode, epoch):
        logger.info(
            "avg_loss: {} ".format(round(stats.xent(), 5)) +
            "words acc: {} ".format(round(100 * (stats.n_correct / stats.n_words), 2)) +
            "lr: {}".format(self.optim_schedule.get_lr())
        )
