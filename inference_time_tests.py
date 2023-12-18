import torch
from learned_index import LearnedIndex
from dataloader import get_dataloader
from utils import get_config


def test1():
    learned_index = LearnedIndex('output_adam/sim1_weights.pt')
    config = get_config()
    dataloader = get_dataloader(config['data_path'], config['batch_size'])
    keys, locations = next(iter(dataloader))
    predictions, time_taken = learned_index.get_predictions(keys)
    print(f"Evaluating {len(keys)} keys took {time_taken}ms - {time_taken/len(keys)}ms per key")


if __name__ == '__main__':
    test1()
