import math
import time
import pickle
import json


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_txt(path):
    with open(path, encoding='UTF-8', errors='ignore') as f:
        data = [i.strip() for i in f.readlines() if len(i) > 0]
    return data


def save_txt(data, path):
    with open(path, 'w', encoding='UTF-8') as f:
        f.write(data)


def load_json(path):
    with open(path, 'r', encoding='UTF_8') as f:
        return json.load(f)


def save_json(data, path, indent=0):
    with open(path, 'w', encoding='UTF-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.start_time = time.time()
        self._reset()

    def _reset(self, keys=None):
        self.steps_loss = 0
        self.steps_words = 0
        if keys:
            for k in keys:
                setattr(self, k, 0)

    def update(self, loss, num_non_padding, metrics):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not
        """
        self.steps_loss += loss
        self.steps_words += num_non_padding
        self.loss += loss
        self.n_words += num_non_padding

        for k, v in metrics.items():
            if not hasattr(self, k):
                if isinstance(v, int) or isinstance(v, float):
                    setattr(self, k, 0)
                elif isinstance(v, list):
                    setattr(self, k, [])
            setattr(self, k, getattr(self, k) + v)

            if not hasattr(self, "step_" + k):
                if isinstance(v, int) or isinstance(v, float):
                    setattr(self, "step_" + k, 0)
                elif isinstance(v, list):
                    setattr(self, "step_" + k, [])
            setattr(self, "step_" + k, getattr(self, "step_" + k) + v)

    # def report(self):
    #     assert self.steps_d_n_correct / self.n_words == (self.tp + self.tn / (self.tp + self.fp + self.tn + self.fn))
    #     prec = self.tp / (self.tp + self.fp)
    #     recall = self.tp / (self.tp + self.fn)
    #     f1 = 2 * (prec * recall) / (prec + recall)
    #     output = {"loss": self.steps_loss / self.steps_words,
    #               "d_acc": 100 * (self.steps_d_n_correct / self.n_words),
    #               "prec": prec,
    #               "recall": recall,
    #               "f1": f1,
    #               "c_acc": 100 * (self.steps_c_n_correct / self.n_words),
    #               "elapsed_time": self.elapsed_time()}
    #     self._reset()
    #     return output

    def aprf(self, k):
        """ compute accuracy precision recall F1 """
        (tp, tn, fp, fn) = (getattr(self, k + attr) for attr in ["tp", "tn", "fp", "fn"])

        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        # assert self.n_correct / self.n_words == acc
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
        return acc, prec, recall, f1

    # def accuracy(self):
    #     """ compute accuracy """
    #     return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    # def output(self, step, num_steps, learning_rate, start):
    #     """Write out statistics to stdout.
    #
    #     Args:
    #        step (int): current step
    #        n_batch (int): total batches
    #        start (int): start time of step.
    #     """
    #     t = self.elapsed_time()
    #     step_fmt = "%2d" % step
    #     if num_steps > 0:
    #         step_fmt = "%s/%5d" % (step_fmt, num_steps)
    #     logger.info(
    #         ("Step %s; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
    #          "lr: %7.7f; %3.0f/%3.0f tok/s; %6.0f sec")
    #         % (step_fmt,
    #            self.accuracy(),
    #            self.ppl(),
    #            self.xent(),
    #            learning_rate,
    #            self.n_src_words / (t + 1e-5),
    #            self.n_words / (t + 1e-5),
    #            time.time() - start))
    #     sys.stdout.flush()
    #
    # def log_tensorboard(self, prefix, writer, learning_rate, step):
    #     """ display statistics to tensorboard """
    #     t = self.elapsed_time()
    #     writer.add_scalar(prefix + "/xent", self.xent(), step)
    #     writer.add_scalar(prefix + "/ppl", self.ppl(), step)
    #     writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
    #     writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
    #     writer.add_scalar(prefix + "/lr", learning_rate, step)
