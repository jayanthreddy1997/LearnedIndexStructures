import time
import torch
import torch.nn as nn
import pandas as pd
import os
import yaml

from model import NeuralNetwork
from dataloader import get_dataloader


N_LAYERS = 2
N_UNITS = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
N_EPOCHS = 100
BATCH_SIZE = 16384
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    losses = []
    start_time = time.time()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    train_time = time.time() - start_time
    avg_loss = sum(losses) / len(losses)
    return avg_loss, train_time


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            test_loss += loss_fn(outputs, y).item()
    test_loss /= num_batches
    return test_loss


def run_simulation(config):
    data_path = config['data_path']
    save_stats = config['save_stats']
    stats_fp = config['stats_fp']
    weights_fp = config['weights_fp']
    model_scripted_fp = config['model_scripted_fp']

    torch.manual_seed(0)
    print("Creating Neural Network")
    model = NeuralNetwork(n_layers=N_LAYERS, n_units=N_UNITS).to(device)
    print("Creating dataloaders")
    dataloader = get_dataloader(data_path, batch_size=BATCH_SIZE)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    stats = []
    start_time = time.time()
    print(f"Starting simulation")
    best_loss = float('inf')
    for epoch_num in range(1, N_EPOCHS + 1):
        train_loss, train_time = train(dataloader, model, loss_fn, optimizer)
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model, weights_fp)
        print(f"Epoch {epoch_num}/{N_EPOCHS} | Train loss: {train_loss} | Train time: {train_time}s")
        stats.append([epoch_num, N_EPOCHS, time.time() - start_time, train_loss])

    model.load_state_dict(torch.load(weights_fp))

    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(model_scripted_fp)  # Save

    model.eval()
    test_loss = test(dataloader, model, loss_fn)
    print(f"Test loss: {test_loss}")
    stats.append(["final", N_EPOCHS, time.time() - start_time, test_loss])
    print(f"Simulation completed in {time.time() - start_time}s")
    stats_df = pd.DataFrame(stats, columns=['EpochNum', 'TotalEpochs', 'TimeElapsed', 'TrainLoss'])
    if save_stats:
        stats_df.to_csv(stats_fp)


if __name__ == '__main__':
    config_file = 'conf/hpc_config.yml' if 'hpc.nyu.edu' in os.uname().nodename else 'conf/local_config.yml'
    with open(config_file, 'r') as f:
        _config = yaml.safe_load(f)
    run_simulation(_config)
