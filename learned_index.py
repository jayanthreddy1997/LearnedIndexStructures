import torch
import torch._dynamo
from model import NeuralNetwork
from utils import get_config
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LearnedIndex:
    def __init__(self, model_fp):
        config = get_config()
        model = NeuralNetwork(n_layers=config['n_layers'], n_units=config['n_units'])
        if device == 'cpu':
            model.load_state_dict(torch.load(model_fp, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_fp, map_location=torch.device('cuda')))
        model.eval()
        model = model.to(device)
        torch._dynamo.reset()
        self.model = model
#         self.model = torch.compile(model)

    def get_predictions(self, keys):
        """

        :param keys:
        :return: predicted locations, time taken in seconds
        """
        keys = keys.to(device)
        self.model.eval()
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            pred = self.model(keys)
            end.record()
            torch.cuda.synchronize()
        return pred, start.elapsed_time(end)

