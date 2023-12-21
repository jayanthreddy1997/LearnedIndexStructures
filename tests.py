import struct
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataloader import get_dataloader
from utils import get_config
from learned_index import LearnedIndex


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_data():
    data_fp = f'/Users/reddyj/Desktop/workspace/nyu/courses/idls/project/SOSD/data/normal_200M_uint32'
    with open(data_fp, 'rb') as f:
        packed_data = f.read()
        num_records = struct.unpack('Q', packed_data[:8])[0]
        data = struct.unpack(f'{num_records}I', packed_data[8:])
    plt.hist(data[200000000-1000000:], bins=100)
    plt.show()


def test_torch_compile():
    import torch

    torch.set_default_device("cuda:0")
    @torch.compile
    def test_fn(x):
      return torch.sin(x)

    a = torch.zeros(100)
    print(test_fn(a))


def test_model_perf():
    """
    Computes the R-squared performance metric
    """
    overall_tss = 0.0
    overall_rss = 0.0
    for dataset in ['norm', 'logn', 'uspr']:
        config = get_config(f'conf/hpc_config_{dataset}.yml')
        dataloader = get_dataloader(config['data_path'], batch_size=100000)
        learned_index = LearnedIndex(config['weights_fp'])

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predicted_values, _ = learned_index.get_predictions(inputs)

            mean_targets = torch.mean(targets)
            tss_minibatch = torch.sum((targets - mean_targets) ** 2)

            # Calculate Residual Sum of Squares (RSS) for the minibatch
            rss_minibatch = torch.sum((targets - predicted_values) ** 2)

            # Accumulate TSS and RSS
            overall_tss += tss_minibatch.item()
            overall_rss += rss_minibatch.item()
        # Compute overall R-squared
        overall_r_squared = 1 - (overall_rss / overall_tss)
        print(f"Dataset: {dataset}, Overall R-squared: {overall_r_squared}")


def test_memory_usage():
    for dataset in ['norm', 'logn', 'uspr']:
        config = get_config(f'conf/hpc_config_{dataset}.yml')
        dataloader = get_dataloader(config['data_path'], batch_size=100000)
        learned_index = LearnedIndex(config['weights_fp'])

        keys, locations = next(iter(dataloader))
        keys = keys.to(device)
        for i in range(5):  # warmup
            predictions, time_taken = learned_index.get_predictions(keys)
        torch.cuda.reset_peak_memory_stats(device=torch.device('cuda'))
        temp1, temp2 = learned_index.get_predictions(keys)
        print(f"Dataset: {dataset} | Peak memory consumption: {torch.cuda.max_memory_allocated(device=torch.device('cuda'))}")


if __name__ == '__main__':
    test_model_perf()
    test_memory_usage()
