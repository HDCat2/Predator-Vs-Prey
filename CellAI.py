import torch.nn as nn
import torch.nn.functional as F

class CellNet(nn.Module):
    INITIAL_MUTATION_RATE = 0.5
    DECAY_RATE = 1
    NUM_HIDDEN_LAYERS = 2 # Minimum of 1 hidden layer, otherwise change init
    NUM_HIDDEN_LAYER_NEURONS = 64
    def __init__(self, rayCount):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(rayCount + 1, CellNet.NUM_HIDDEN_LAYER_NEURONS, bias=False))
        for i in range(CellNet.HIDDEN_LAYERS - 1):
            self.layers.append(nn.Linear(CellNet.NUM_HIDDEN_LAYER_NEURONS, CellNet.NUM_HIDDEN_LAYER_NEURONS, bias=False))
        self.layers.append(nn.Linear(CellNet.NUM_HIDDEN_LAYER_NEURONS, 2, False)) # Output layer corresponds to speed and angular velocity

        self.eval() # With our method, we won't need to train the model

    def forward(self, x):
        """ Feed """
        for i in range(CellNet.NUM_HIDDEN_LAYERS):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)

    def mutate(self, generation):
        decay = 1/(1 + generation * CellNet.DECAY_RATE) * CellNet.INITIAL_MUTATION_RATE
        raise NotImplementedError()