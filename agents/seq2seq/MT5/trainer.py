import os
import tqdm
import logging
import platform

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Tokenizer, MT5Config, MT5ForConditionalGeneration

from utils import Statistics
from agents.trainer_base import BaseTrainer
from agents.optim_schedule import ScheduledOptim, _get_optimizer
from .data_utils import build_dataset, collate

logger = logging.getLogger(__file__)


class Trainer(BaseTrainer):

    @classmethod
    def add_cmdline_args(cls, argparser):
        super(Trainer, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('MT5 Arguments')
        # add_common_cmdline_args(agent)
        # memory and knowledge arguments

        agent.add_argument('--checkpoint', type=str, default="google/mt5-small")

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
        _optimizer = _get_optimizer(self.model, opt)
        self.optim_schedule = ScheduledOptim(opt, _optimizer)

        self.skip_report_eval_steps = opt.skip_report_eval_steps

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
            generated = self.model.generate(input_ids, attention_mask=input_mask)
            dec = self.tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
            input_ids, input_mask, labels = tuple(
                input_tensor.to(self.device) for input_tensor in batch)

            outputs = self.model(input_ids, attention_mask=input_mask, labels=labels,
                                 return_dict=True)  # prob [batch_size, seq_len, 1]
            loss, logits = outputs.loss, outputs.logits

            generated = None
            if data_type != "train" and epoch > self.skip_report_eval_steps:
                generated = self.model.generate(input_ids, attention_mask=input_mask, max_length=labels.size(1) + 1)
                generated = generated[:, 1:]
                if generated.size(1) < labels.size(1):
                    generated = pad_sequence([labels[0]] + [one for one in generated], batch_first=True,
                                             padding_value=self.tokenizer.pad_token_id)[1:]

            if data_type == "train":
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward(retain_graph=True)
                if step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)
                    self.optim_schedule.step()
                    self.optim_schedule.zero_grad()

            # sta
            # self._stats(stats, loss.item(), logits.softmax(dim=-1).argmax(dim=-1), labels)
            self._stats(stats, loss.item(), generated, labels)
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
        self._report(stats, data_type, epoch)
        # return round(stats.xent(), 5)
        return round(100 * 2 * stats.TP / (2 * stats.TP + stats.FN + stats.FP),
                     4) if data_type != "train" and epoch > self.skip_report_eval_steps else -round(
            stats.xent(), 5)

    def _stats(self, stats: Statistics, loss, preds, target):
        non_padding = target.ne(-100)
        num_non_padding = non_padding.sum().item()
        if preds is None:
            stats.update(loss * num_non_padding, num_non_padding, {})
            return

        preds_dec = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        target_forgen = target.clone()
        target_forgen[target == -100] = 0
        labels_dec = self.tokenizer.batch_decode(target_forgen, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
        if self.show_case:
            for i in range(5):
                logger.info("pred: {} ".format(preds_dec[i]))
                logger.info("label: {} ".format(labels_dec[i]))
                logger.info("--------------------------------------")
            self.show_case = False
        assert len(preds_dec) == len(labels_dec)
        total_utter_number = 0
        correct_utter_number = 0
        TP, FP, FN = 0, 0, 0
        for pred_utterance, anno_utterance in zip(preds_dec, labels_dec):
            anno_semantics = [one.split("-") for one in anno_utterance.split(";")]
            pred_semantics = [one.split("-") for one in pred_utterance.split(";")]
            anno_semantics = set([tuple(item) for item in anno_semantics])
            pred_semantics = set([tuple(item) for item in pred_semantics])

            total_utter_number += 1
            if anno_semantics == pred_semantics:
                correct_utter_number += 1

            TP += len(anno_semantics & pred_semantics)
            FN += len(anno_semantics - pred_semantics)
            FP += len(pred_semantics - anno_semantics)

        metrics = {
            "n_correct": preds.eq(target).masked_select(non_padding).sum().item(),
            "n_correct_utt": sum(x.eq(y).masked_select(z).all().float().item()
                                 for x, y, z in zip(preds, target, non_padding)),
            "n_utterances": target.size(0),
            "TP": TP,
            "FN": FN,
            "FP": FP,
            "correct_utter_number": correct_utter_number,
            "total_utter_number": total_utter_number
            # "d_tp": (preds.eq(1) & target.eq(1)).masked_select(non_padding).sum().item(),
            # "d_fp": (preds.eq(1) & target.eq(0)).masked_select(non_padding).sum().item(),
            # "d_tn": (preds.eq(0) & target.eq(0)).masked_select(non_padding).sum().item(),
            # "d_fn": (preds.eq(0) & target.eq(1)).masked_select(non_padding).sum().item(),
        }
        stats.update(loss * num_non_padding, num_non_padding, metrics)

    def _report(self, stats: Statistics, mode, epoch):
        if mode == "train" and epoch <= self.skip_report_eval_steps:
            logger.info("avg_loss: {} ".format(round(stats.xent(), 5)))
        else:
            logger.info(
                "avg_loss: {} ".format(round(stats.xent(), 5)) +
                "words acc: {} ".format(round(100 * (stats.n_correct / stats.n_words), 2)) +
                "utterances acc: {} ".format(round(100 * (stats.n_correct_utt / stats.n_utterances), 2)) +
                "Precision %.2f " % (100 * stats.TP / (stats.TP + stats.FP)) +
                "Recall %.2f " % (100 * stats.TP / (stats.TP + stats.FN)) +
                "F1-score %.2f " % (100 * 2 * stats.TP / (2 * stats.TP + stats.FN + stats.FP)) +
                "Joint accuracy %.2f " % (100 * stats.correct_utter_number / stats.total_utter_number) +
                "lr: {}".format(self.optim_schedule.get_lr())
            )
