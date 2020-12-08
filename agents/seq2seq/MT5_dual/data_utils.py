import os
import logging
import torch
import collections
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__file__)


def build_dataset(dataset, tokenizer, name):
    logger.info("Tokenize and encode the dataset {} ".format(name))
    instances = collections.defaultdict(list)
    for line in dataset:
        x = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line[0]))
        y = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line[1]))
        input_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("query: ")) + x
        label_idx = y + [tokenizer.eos_token_id]

        input_mask = [1 for _ in range(len(input_idx))]

        instances["pad_input"].append(input_idx)
        instances["pad_input_mask"].append(input_mask)
        instances["pad_label"].append(label_idx)

        reverse_input_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("semantic: ")) + y
        reverse_label_idx = x + [tokenizer.eos_token_id]

        reverse_input_mask = [1 for _ in range(len(reverse_input_idx))]

        instances["pad_reverse_input"].append(reverse_input_idx)
        instances["pad_reverse_input_mask"].append(reverse_input_mask)
        instances["pad_reverse_label"].append(reverse_label_idx)

    return instances


def collate(dataset, pad_id, name, batch_first=True):
    logger.info("Pad inputs and convert to Tensor {} ".format(name))
    tensor_dataset = []
    for input_name in dataset.keys():
        if "pad" in input_name:
            if "label" in input_name in input_name:
                input_tensor = pad_sequence(
                    [torch.tensor(feature, dtype=torch.long) for feature in dataset[input_name]],
                    batch_first=batch_first, padding_value=-100)
            else:
                input_tensor = pad_sequence(
                    [torch.tensor(feature, dtype=torch.long) for feature in dataset[input_name]],
                    batch_first=batch_first, padding_value=pad_id)
        else:
            input_tensor = torch.tensor(dataset[input_name], dtype=torch.long)
        tensor_dataset.append(input_tensor)
    logging.info("Max len of input tensor is %d" % tensor_dataset[0].shape[1])
    logging.info("Max len of label tensor is %d" % tensor_dataset[-1].shape[1])
    return tensor_dataset


class BertDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_len=512, pad_first=True, mode='train'):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_len = max_len
        self.data_size = len(dataset)
        self.pad_first = pad_first
        self.mode = mode

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        item = self.dataset.iloc[item]
        input_ids = item['random_text']
        input_ids = ['[CLS]'] + list(input_ids)[:min(len(input_ids), self.max_len - 2)] + ['[SEP]']
        # convert to bert ids
        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        pad_len = self.max_len - len(input_ids)
        if self.pad_first:
            input_ids = [0] * pad_len + input_ids
            input_mask = [0] * pad_len + input_mask
            segment_ids = [0] * pad_len + segment_ids
        else:
            input_ids = input_ids + [0] * pad_len
            input_mask = input_mask + [0] * pad_len
            segment_ids = segment_ids + [0] * pad_len

        output = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        }

        if self.mode == 'train':
            output_ids = item['origin_text']
            label = item['label']
            label = [int(x) for x in label if x != ' ']
            output_ids = ['[CLS]'] + list(output_ids)[:min(len(output_ids), self.max_len - 2)] + ['[SEP]']
            label = [0] + label[:min(len(label), self.max_len - 2)] + [0]

            output_ids = self.tokenizer.convert_tokens_to_ids(output_ids)
            pad_label_len = self.max_len - len(label)
            if self.pad_first:
                output_ids = [0] * pad_len + output_ids
                label = [0] * pad_label_len + label
            else:
                output_ids = output_ids + [0] * pad_len
                label = label + [0] * pad_label_len

            output = {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'output_ids': output_ids,
                'label': label
            }
        return {key: torch.tensor(value) for key, value in output.items()}
