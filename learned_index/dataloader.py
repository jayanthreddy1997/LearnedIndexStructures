import struct
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class LearnedIndexDataset(Dataset):
    def __init__(self, data_fp):
        data = get_data(data_fp)[:100000]
        label = np.arange(0, len(data), dtype=np.float32)

        self.data = np.asarray((data - np.mean(data))/np.std(data), dtype=np.float32).reshape((len(data), 1))
        self.label = np.asarray((label - np.mean(label))/np.std(label), dtype=np.float32).reshape((len(label), 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def get_data(file_path):
    with open(file_path, 'rb') as f:
        packed_data = f.read()
        num_records = struct.unpack('Q', packed_data[:8])[0]
        data = struct.unpack(f'{num_records}I', packed_data[8:])
        # TODO: extend to other data types
    return np.asarray(data, dtype=np.uint32)


def get_dataloader(file_path, batch_size=512):
    torch.manual_seed(0)

    dataset_ = LearnedIndexDataset(file_path)
    return DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)


if __name__ == '__main__':
    fp = f'/Users/reddyj/Desktop/workspace/nyu/courses/idls/project/SOSD/data/normal_200M_uint32'
    dataloader = get_dataloader(fp, batch_size=2)
    print(len(dataloader))
    print(next(iter(dataloader)))


