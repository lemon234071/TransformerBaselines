import os
import random


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
        with open(path, encoding='UTF-8', errors='ignore') as f:
            datasets[name] = [i.strip() for i in f.readlines() if len(i) > 0]
    return datasets


def extrac_data(path):
    with open(path, encoding="utf-8") as f:
        data = [x.strip() for x in f.readlines() if len(x) > 0]
    data_pos = []
    data_neg = []
    for line in data:
        seq = line.split("\t")
        if seq[-1] == "0":
            if seq[1] != "NULL":
                if len(seq[0]) == len(seq[1]):
                    data_neg.append("\t".join(seq))
            else:
                seq[1] = seq[0]
                data_pos.append("\t".join(seq))
    data_neg = list(set(data_neg))
    print("neg len", len(data_neg))
    random.shuffle(data_neg)
    neg_train = data_neg[:int(len(data_neg) * 0.8)]
    neg_valid = data_neg[int(len(data_neg) * 0.8): int(len(data_neg) * 0.9)]
    neg_test = data_neg[int(len(data_neg) * 0.9):]

    data_pos = list(set(data_pos))
    print("pos len", len(data_pos))
    random.shuffle(data_pos)
    pos_train = data_pos[:int(len(data_pos) * 0.8)]
    pos_valid = data_pos[int(len(data_pos) * 0.8): int(len(data_pos) * 0.9)]
    pos_test = data_pos[int(len(data_pos) * 0.9):]

    train = neg_train + pos_train
    valid = neg_valid + pos_valid
    test = neg_test + pos_test
    for k, v in {"train": train, "valid": valid, "test": test}.items():
        dirpath = "data/xiaowei/all/"
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        print(k, "len", len(v))
        with open(dirpath + k + ".txt", "w", encoding="utf-8") as f:
            f.write("\n".join(v))

    for k, v in {"neg_train": neg_train, "neg_valid": neg_valid, "neg_test": neg_test}.items():
        dirpath = "data/xiaowei/neg/"
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        print(k, "len", len(v))
        with open(dirpath + k + ".txt", "w", encoding="utf-8") as f:
            f.write("\n".join(v))


# if __name__ == '__main__':
#     extrac_data("TransformerBaselines/data/raw/xiaowei.txt")
