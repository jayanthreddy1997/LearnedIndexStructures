from learned_index import LearnedIndex
from dataloader import get_dataloader
import numpy as np
from utils import get_config

NUM_QUERIES = 100000


def test_inference_time():
    for dataset in ['norm', 'logn', 'uspr']:
        config = get_config(f'conf/hpc_config_{dataset}.yml')
        learned_index = LearnedIndex(config['weights_fp'])
        dataloader = get_dataloader(config['data_path'], NUM_QUERIES)
        keys, locations = next(iter(dataloader))
        for i in range(5):
            predictions, time_taken = learned_index.get_predictions(keys)
        times = [learned_index.get_predictions(keys)[1] for _ in range(10)]
        avg_time = np.mean(times)
        print(f"Dataset: {dataset} - Evaluating {len(keys)} keys took {avg_time}ms")


if __name__ == '__main__':
    test_inference_time()
