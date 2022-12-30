import torch.nn as nn
import torch.nn.functional as F

class CellNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fx1(x))
        x = F.relu(self.fx2(x))
        x = self.fx3(x)
        return F.log_softmax(x, dim=1)

    def mutate(self, generation):
        pass