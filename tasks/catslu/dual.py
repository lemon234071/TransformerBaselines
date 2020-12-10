import os
import json
import logging
import collections

logger = logging.getLogger(__file__)


def get_datasets(dir_path):
    datasets = {}
    for file in os.listdir(dir_path):
        name = None
        for x in ["train", "valid", "dev", "test"]:
            if x in file:
                name = x
        if not name:
            continue

        path = os.path.join(dir_path, file)
        if "json" in path:
            with open(path, encoding='UTF-8') as f:
                datasets[name] = json.load(f)
        else:
            with open(path, encoding='UTF-8', errors='ignore') as f:
                datasets[name] = [i.strip() for i in f.readlines() if len(i) > 0]
    return datasets


def build_dataset(name, dataset, tokenizer):
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
