import torch.nn as nn
from config import Config

class ProductRecommender(nn.Module):
    def __init__(self, input_dim: int, num_products: int):
        super(ProductRecommender, self).__init__()
        from config import Config
        self.layers = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM * 2, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, num_products),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)