import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np
import pandas as pd
import os

class PurchaseBinaryDataset(Dataset):
    """
    PyTorch Dataset for the binary classification task.
    Each sample consists of:
      - A list of purchased product names (strings).
      - A binary label: 1 if user purchased anything after cutoff, 0 otherwise.
    """
    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): Must contain columns:
                'products_before' -> list of product names
                'label_binary'    -> integer (0 or 1)
        """
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # products_before is a list of strings
        products = row['products_before']  
        # binary label (0 or 1)
        label = row['order_after']        
        return products, torch.tensor(label, dtype=torch.float)

class PurchaseMultiLabelDataset(Dataset):
    """
    PyTorch Dataset for the multi-label product prediction task.
    Each sample consists of:
      - A list of purchased product names (strings).
      - A multi-hot label vector for candidate products.
    """
    def __init__(self, df, candidate_product_list):
        """
        Args:
            df (pd.DataFrame): Must contain columns:
                'products_before' -> list of product names
                'products_after'  -> list of product names purchased after cutoff
            candidate_product_list (list of str): fixed list of candidate products
        """
        self.data = df.reset_index(drop=True)
        self.candidate_product_list = candidate_product_list
        # Create a mapping product -> index
        self.product2idx = {prod: i for i, prod in enumerate(candidate_product_list)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        products_before = row['products_before']  # list of strings
        products_after  = row['post_cutoff']   # list of strings

        # Create a multi-hot label vector of shape [len(candidate_product_list)]
        # 1 if product appears in products_after, else 0
        label_vec = torch.zeros(len(self.candidate_product_list), dtype=torch.float)
        for p in products_after:
            if p in self.product2idx:  # only set if p is known in candidate list
                label_vec[self.product2idx[p]] = 1.0
        
        return products_before, label_vec
    