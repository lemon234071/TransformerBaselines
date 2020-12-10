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
        input_idx = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("semantic: " + line[1]))
        label_idx = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(line[0]))
        input_seq = input_idx
        label_seq = label_idx + [tokenizer.eos_token_id]

        input_mask = [1 for _ in range(len(input_seq))]

        instances["pad_input"].append(input_seq)
        instances["pad_input_mask"].append(input_mask)
        instances["pad_label"].append(label_seq)

    return instances
