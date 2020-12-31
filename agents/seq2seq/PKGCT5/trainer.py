import json
import tqdm
import logging
import platform

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Tokenizer, MT5Config, MT5ForConditionalGeneration

from agents.utils import Statistics
from agents.trainer_base import BaseTrainer
from agents.data_utils import collate
from .model import PKGCT5mayi

logger = logging.getLogger(__file__)


class Trainer(BaseTrainer):

    @classmethod
    def add_cmdline_args(cls, argparser):
        super(Trainer, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('MT5 Arguments')
        # add_common_cmdline_args(agent)
        # memory and knowledge arguments

        agent.add_argument('--checkpoint', type=str, default="google/mt5-small")
        agent.add_argument('--dual', type=bool, default=False)
        agent.add_argument('--da', type=bool, default=False)
        agent.add_argument('--beam', type=int, default=1)
        agent.add_argument('--graph_dir', type=str, default="")

    def __init__(self, opt, device):
        super(Trainer, self).__init__(opt, device)
        self.tokenizer = T5Tokenizer.from_pretrained(opt.vocab_path
                                                     if opt.vocab_path else opt.checkpoint, do_lower_case=True)
        self.config = MT5Config.from_pretrained(opt.checkpoint)

        self.model = PKGCT5mayi(self.config).to(device) \
            if platform.system() == 'Windows' else \
            PKGCT5mayi.from_pretrained(opt.checkpoint, config=self.config).to(device)
        # raise Exception("handle the embedding")
        # if os.path.isdir(opt.checkpoint):
        #     self.model.
        # if torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for train" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=[0,1,2])

        self.skip_report_eval_steps = opt.skip_report_eval_steps
        self.dual = opt.dual
        self.da = opt.da
        self.beam = opt.beam

        if opt.graph_dir:
            self.load_graph(opt.graph_dir)

    def load_data(self, data_type, dataset, build_dataset, infer=False):
        self.dataset[data_type] = build_dataset(data_type, dataset, self.tokenizer)
        tensor_dataset = collate(self.dataset[data_type], self.tokenizer.pad_token_id, data_type)
        dataset = TensorDataset(*tensor_dataset)
        self._dataloader[data_type] = DataLoader(dataset,
                                                 batch_size=self.opt.batch_size,
                                                 num_workers=self.opt.num_workers,
                                                 shuffle=(data_type == "train" and not infer))

    def load_graph(self, graph_dir):
        # node2idx
        nodes_vocab = {}
        with open("%s/nodes_vocab.txt" % (graph_dir), encoding="utf-8") as f:
            for i, line in enumerate(f):
                nodes_vocab.setdefault(line.strip(), i)
        # special for t5
        # nodes_vocab.setdefault("<pad>", 0)
        # nodes_vocab.setdefault("▁", 0)

        # init adj_mat
        np_adj_mat = np.zeros([len(nodes_vocab), len(nodes_vocab)], dtype=np.float)

        # load Graphs containing all types
        with open("%s/spellGraphs.txt" % (graph_dir), encoding="utf-8") as f:
            for i, line in enumerate(f):
                e1, e2, rel = line.strip().split("|")
                if rel in ["近音异调", "同音异调"]:
                    np_adj_mat[nodes_vocab[e1], nodes_vocab[e2]] = 1
                    np_adj_mat[nodes_vocab[e2], nodes_vocab[e1]] = 1

        # model vocab
        # build word2node (ct5_vocab's word (idx) to nodes_vocab's n)
        w2n = []
        vocab = self.tokenizer.get_vocab()
        node_notin_vocab = [x for x in nodes_vocab if x not in vocab]
        print(len(node_notin_vocab), "nodes not in vocab")
        for i, (word, idx) in enumerate(vocab.items()):
            if word in nodes_vocab:
                w2n.append(nodes_vocab[word])
            elif word.replace("▁", "") in nodes_vocab:
                w2n.append(nodes_vocab[word.replace("▁", "")])
            else:
                w2n.append(0)
        # vocab = {}
        # with open("%s/ct5_vocab.txt" % (graph_dir)) as f:
        #     for i, line in enumerate(f):
        #         word = line.strip()
        #         vocab.setdefault(word, i)
        #         if word in nodes_vocab:
        #             w2n.append(nodes_vocab[word])
        #         else:
        #             w2n.append(0)
        # build node2word (nodes_vocab's n to ct5_vocab's word idx)

        # TODO(yida) check the process of "▁您"
        n2w = []
        with open("%s/nodes_vocab.txt" % (graph_dir), encoding="utf-8") as f:
            for i, line in enumerate(f):
                word = line.strip()
                if word in vocab:
                    n2w.append(vocab[word])
                else:
                    n2w.append(0)

        def norm_adj(adj):
            adj = adj + torch.eye(adj.size(0)).to(adj.device)
            D = torch.sum(adj, dim=-1) + 1e-8
            D = torch.pow(D, -0.5)
            D = torch.diag(D)
            # matmul, mm, bmm
            adj = torch.mm(adj, D)
            adj = torch.mm(adj.t(), D)
            # D = tf.pow(tf.reduce_sum(_adj_mat, axis=2) + 1e-8, -0.5)
            # D = tf.matrix_diag(D)
            # _adj_mat = tf.keras.backend.batch_dot(_adj_mat, D)
            # # print(_adj_mat)
            # _adj_mat = tf.transpose(_adj_mat, [0, 2, 1])
            # _adj_mat = tf.keras.backend.batch_dot(_adj_mat, D)
            return adj

        self.adj_mat = norm_adj(torch.from_numpy(np_adj_mat).to(self.device))
        self.w2n, self.n2w = torch.tensor(w2n).to(self.device), torch.tensor(n2w).to(self.device)
        return

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
            input_ids, input_mask, labels, r_input_ids, r_input_mask, r_labels = tuple(
                input_tensor.to(self.device) for input_tensor in batch)
            generated = self.model.generate(input_ids, attention_mask=input_mask, max_length=150,
                                            num_beams=self.beam, num_return_sequences=self.beam,
                                            adj=self.adj_mat, w2n=self.w2n, n2w=self.n2w)
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
            input_ids, input_mask, labels, r_input_ids, r_input_mask, r_labels = batch
            input_ids, input_mask, labels = tuple(x.to(self.device) for x in [input_ids, input_mask, labels])
            # tuple(input_tensor.to(self.device) for input_tensor in batch)

            # forward
            outputs = self.model(input_ids, attention_mask=input_mask, labels=labels,
                                 return_dict=True, adj=self.adj_mat, w2n=self.w2n,
                                 n2w=self.n2w)  # prob [batch_size, seq_len, 1]
            loss, logits = outputs.loss, outputs.logits

            # reverse forward
            if self.dual:
                r_input_ids, r_input_mask, r_labels = tuple(
                    x.to(self.device) for x in [r_input_ids, r_input_mask, r_labels])
                reverse_outputs = self.model(r_input_ids, attention_mask=r_input_mask, labels=r_labels,
                                             return_dict=True)  # prob [batch_size, seq_len, 1]
                reverse_loss = reverse_outputs.loss
                loss = loss + reverse_loss

            # data augment
            if data_type == "train" and self.da:
                r_input_ids, r_input_mask, r_labels = tuple(
                    x.to(self.device) for x in [r_input_ids, r_input_mask, r_labels])
                da_input_ids, da_input_mask, da_labels = self.get_da(r_input_ids, r_input_mask, r_labels)
                da_outputs = self.model(da_input_ids, attention_mask=da_input_mask, labels=da_labels,
                                        return_dict=True)  # prob [batch_size, seq_len, 1]
                loss = loss + da_outputs.loss

            if data_type == "train":
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()
                if step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)
                    self.optim_schedule.step()
                    self.optim_schedule.zero_grad()

            # sta
            generated = self.eval_gen(data_type, epoch, input_ids, input_mask, labels)
            self._stats(stats, loss.item(), generated, labels)

        logger.info("Epoch{}_{}, ".format(epoch, str_code))

        self._report(stats, data_type, epoch)
        return self.metric(data_type, epoch, stats)

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
        self.tokenizer.batch_decode(target_forgen[:, 3:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # if self.show_case:
        #     for i in range(5):
        #         logger.info("pred: {} ".format(preds_dec[i]))
        #         logger.info("label: {} ".format(labels_dec[i]))
        #         logger.info("--------------------------------------")
        #     self.show_case = False

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
        }
        stats.update(loss * num_non_padding, num_non_padding, metrics)

    def _report(self, stats: Statistics, mode, epoch):
        if mode == "train" or epoch <= self.skip_report_eval_steps:
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

    def eval_gen(self, data_type, epoch, input_ids, input_mask, labels):
        if data_type != "train" and epoch > self.skip_report_eval_steps:
            self.model.eval()
            generated = self.model.generate(input_ids, attention_mask=input_mask, adj=self.adj_mat, w2n=self.w2n,
                                            n2w=self.n2w, max_length=labels.size(1) + 1)
            generated = generated[:, 1:]
            if generated.size(1) < labels.size(1):
                generated = pad_sequence([labels[0]] + [one for one in generated], batch_first=True,
                                         padding_value=self.tokenizer.pad_token_id)[1:]
            self.model.train()
            return generated
        else:
            return None

    def metric(self, data_type, epoch, stats):
        if data_type != "train" and epoch > self.skip_report_eval_steps:
            return round(100 * 2 * stats.TP / (2 * stats.TP + stats.FN + stats.FP), 4)
        else:
            return -round(stats.xent(), 5)

    def get_da(self, r_input_ids, r_input_mask, r_labels):
        r_generated = self.model.generate(r_input_ids,
                                          attention_mask=r_input_mask,
                                          max_length=r_labels.size(1) + 10,
                                          num_beams=self.beam,
                                          num_return_sequences=self.beam)
        r_generated = r_generated[:, 1:]
        bos_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("query: "))
        bos_tensor = torch.tensor(bos_idx).to(self.device)
        bos_tensor = bos_tensor.repeat(r_generated.size(0), 1)
        da_input_ids = torch.cat([bos_tensor, r_generated], dim=1)
        da_input_mask = (da_input_ids > 0).long()
        da_labels = torch.cat([x.repeat(self.beam, 1) for x in r_input_ids[:, 3:]])
        return da_input_ids, da_input_mask, da_labels
