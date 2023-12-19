import torch
from learned_index import LearnedIndex
from dataloader import get_dataloader
from utils import get_config
import numpy as np

def test1():
    learned_index = LearnedIndex('output_adam/sim1_weights.pt')
    config = get_config()
    dataloader = get_dataloader(config['data_path'], config['batch_size'])
    keys, locations = next(iter(dataloader))
    for i in range(5):
        predictions, time_taken = learned_index.get_predictions(keys)
    times = [learned_index.get_predictions(keys)[1] for _ in range(10)]
    avg_time = np.mean(times)
    print(f"Evaluating {len(keys)} keys took {avg_time}ms - {avg_time/len(keys)}ms per key")


if __name__ == '__main__':
    test1()
