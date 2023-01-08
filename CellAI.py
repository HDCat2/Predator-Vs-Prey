import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rnd
from scipy import special

class CellNet(nn.Module):
    INITIAL_MUTATION_RATE = 1
    DECAY_RATE = 1
    NUM_HIDDEN_LAYERS = 2 # Minimum of 1 hidden layer, otherwise change init
    NUM_HIDDEN_LAYER_NEURONS = 16
    def __init__(self, rayCount):
        super().__init__()
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"
        self.layers = nn.ModuleList().to(self.device)
        self.layers.append(nn.Linear(rayCount + 1, CellNet.NUM_HIDDEN_LAYER_NEURONS, bias=False).to(self.device))
        for i in range(CellNet.NUM_HIDDEN_LAYERS - 1):
            self.layers.append(nn.Linear(CellNet.NUM_HIDDEN_LAYER_NEURONS, CellNet.NUM_HIDDEN_LAYER_NEURONS, bias=False).to(self.device))
        self.layers.append(nn.Linear(CellNet.NUM_HIDDEN_LAYER_NEURONS, 2, False).to(self.device)) # Output layer corresponds to speed and angular velocity

        for layer in self.layers:
            nn.init.constant_(layer.weight, 0)

        self.eval() # With our method, we won't need to train the model

    def forward(self, x, viewDistance):
        """ Feed input into the neural network and obtain movement information as output """
        x = torch.FloatTensor([1 - i/viewDistance for i in x]).to(self.device)
        with torch.no_grad():
            for i in range(CellNet.NUM_HIDDEN_LAYERS):
                x = F.relu(self.layers[i](x)).to(self.device)
            x = self.layers[-1](x).to(self.device)
        return special.expit(x.cpu()).tolist()

    def mutate(self, generation):
        """ Randomly mutate the neural network according to generation number """
        decay = 1/(1 + generation * CellNet.DECAY_RATE) * CellNet.INITIAL_MUTATION_RATE

        with torch.no_grad():
            for k in range(CellNet.NUM_HIDDEN_LAYERS+1):
                curTensor = (self.layers[k].weight.data)
                curTensor = curTensor.tolist()

                for i in range(len(curTensor)):
                    for j in range(int(len(curTensor[i]) * decay)):
                        edge = rnd.randint(0, len(curTensor[i]) - 1)

                        curTensor[i][edge] = special.logit(curTensor[i][edge])
                        curTensor[i][edge] += (rnd.random()-0.5) * decay
                        curTensor[i][edge] = special.expit(curTensor[i][edge])

                self.layers[k].weight.data = torch.FloatTensor(curTensor).to(self.device)

            #self.linear[0].bias.data = torch.zeros((5,), requires_grad=True)  # bias is not a scalar here