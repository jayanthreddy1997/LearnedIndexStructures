import time
import torch
import torch.nn as nn
import pandas as pd
from utils import get_config

from model import NeuralNetwork
from dataloader import get_dataloader


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

    torch.manual_seed(0)
    print("Creating Neural Network")
    model = NeuralNetwork(n_layers=config['n_layers'], n_units=config['n_units']).to(device)
    print("Creating dataloaders")
    dataloader = get_dataloader(config['data_path'], batch_size=config['batch_size'])

    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.MSELoss()
    n_epochs = config['n_epochs']
    stats = []
    start_time = time.time()
    print(f"Starting simulation")
    best_loss = float('inf')
    for epoch_num in range(1, n_epochs + 1):
        train_loss, train_time = train(dataloader, model, loss_fn, optimizer)
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), config['weights_fp'])
        print(f"Epoch {epoch_num}/{n_epochs} | Train loss: {train_loss} | Train time: {train_time}s")
        stats.append([epoch_num, n_epochs, time.time() - start_time, train_loss])

    model.load_state_dict(torch.load(config['weights_fp']))

    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(config['model_scripted_fp'])  # Save

    model.eval()
    test_loss = test(dataloader, model, loss_fn)
    print(f"Test loss: {test_loss}")
    stats.append(["final", n_epochs, time.time() - start_time, test_loss])
    print(f"Simulation completed in {time.time() - start_time}s")
    stats_df = pd.DataFrame(stats, columns=['EpochNum', 'TotalEpochs', 'TimeElapsed', 'TrainLoss'])
    if config['save_stats']:
        stats_df.to_csv(config['stats_fp'])


if __name__ == '__main__':
    run_simulation(get_config())
