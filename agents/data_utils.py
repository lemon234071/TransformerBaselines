import logging
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__file__)


def collate(dataset, pad_id, data_type, batch_first=True):
    logger.info("Pad inputs and convert to Tensor {} ".format(data_type))
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
