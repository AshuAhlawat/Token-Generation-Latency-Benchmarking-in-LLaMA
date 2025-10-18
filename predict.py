import torch
import numpy as np
import torch.nn as nn

mean = np.array([300.2, 499])
std = np.array([811.1,631.3])

network = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.ReLU()
)

network.load_state_dict(torch.load('latency_model.pth'))

token_in = 100
token_out = 10
device = "M3"

tokens = np.array([token_in, token_out])
tokens = (tokens - mean)/std
tokens = tokens.astype(np.float32)

devices = np.array(["M3", "RTX4060", "Ultra9-185H"])
device_arr = (devices == device).astype(np.float32)

arr = torch.from_numpy(np.concatenate((tokens, device_arr)))

print(network(arr).item())

