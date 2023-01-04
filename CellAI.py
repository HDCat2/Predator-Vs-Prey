import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rnd
import math

def sigmoid(x):
    return 2 / (1 + math.e ** -x) - 1

def sigmoidInv(x):
    return -math.log((2 / (x + 1 + 1e-8)) + 1e-7 - 1, math.e)

class CellNet(nn.Module):
    INITIAL_MUTATION_RATE = 100
    DECAY_RATE = 1
    NUM_HIDDEN_LAYERS = 2 # Minimum of 1 hidden layer, otherwise change init
    NUM_HIDDEN_LAYER_NEURONS = 64
    def __init__(self, rayCount):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(rayCount + 1, CellNet.NUM_HIDDEN_LAYER_NEURONS, bias=False))
        for i in range(CellNet.NUM_HIDDEN_LAYERS - 1):
            self.layers.append(nn.Linear(CellNet.NUM_HIDDEN_LAYER_NEURONS, CellNet.NUM_HIDDEN_LAYER_NEURONS, bias=False))
        self.layers.append(nn.Linear(CellNet.NUM_HIDDEN_LAYER_NEURONS, 2, False)) # Output layer corresponds to speed and angular velocity

        for layer in self.layers:
            nn.init.constant_(layer.weight, 0.5)

        self.eval() # With our method, we won't need to train the model

    def forward(self, x):
        """ Feed input into the neural network and obtain movement information as output """
        x = torch.FloatTensor(x)
        with torch.no_grad():
            for i in range(CellNet.NUM_HIDDEN_LAYERS):
                x = F.relu(self.layers[i](x))
            x = self.layers[-1](x)
        return [sigmoid(val) for val in x.tolist()]

    def mutate(self, generation):
        """ Randomly mutate the neural network according to generation number """
        decay = 1/(1 + generation * CellNet.DECAY_RATE) * CellNet.INITIAL_MUTATION_RATE

        with torch.no_grad():
            for k in range(CellNet.NUM_HIDDEN_LAYERS+1):
                curTensor = (self.layers[k].weight.data)
                curTensor = curTensor.tolist()

                for i in range(len(curTensor)):
                    for j in range(len(curTensor[i])):
                        if rnd.random() < decay:

                            curTensor[i][j] = sigmoidInv(curTensor[i][j])
                            curTensor[i][j] *= (1 + (rnd.random()-0.5) * decay)
                            curTensor[i][j] = sigmoid(curTensor[i][j])

                self.layers[k].weight.data = torch.FloatTensor(curTensor)

            #self.linear[0].bias.data = torch.zeros((5,), requires_grad=True)  # bias is not a scalar here

