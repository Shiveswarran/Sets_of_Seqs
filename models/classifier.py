import torch.nn as nn
from config import Config

class WillBuyClassifier(nn.Module):
    def __init__(self, input_dim: int = None):
        super(WillBuyClassifier, self).__init__()
        from config import Config
        input_dim = input_dim or Config.EMBEDDING_DIM
        self.layers = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT / 2),
            nn.Linear(Config.HIDDEN_DIM // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)