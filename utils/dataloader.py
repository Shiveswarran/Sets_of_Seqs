import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from config import Config

class PurchaseDataset(Dataset):
    def __init__(self, df, product_to_idx: dict = None, max_seq_len: int = 50):
        self.df = df
        self.product_to_idx = product_to_idx
        self.max_seq_len = max_seq_len  # Maximum sequence length for padding/truncation
        
        # Extract products_before and convert to lists
        self.products_before = df['products_before'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        ).tolist()
        
        # Extract will_buy labels
        self.will_buy = df['order_after'].values.astype(np.float32)
        
        # Extract next purchases if available
        self.next_purchases = None
        if 'post_cutoff' in df.columns:
            self.next_purchases = df['post_cutoff'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            ).tolist()
        
    def __len__(self):
        return len(self.df)
    
    def pad_sequence(self, sequence: List[int]) -> List[int]:
        """Pad or truncate the sequence to max_seq_len."""
        if len(sequence) > self.max_seq_len:
            return sequence[:self.max_seq_len]
        return sequence + [0] * (self.max_seq_len - len(sequence))  # Padding with 0
    
    def __getitem__(self, idx):
        products = self.products_before[idx]
        will_buy = self.will_buy[idx]
        
        # Convert products to indices and pad the sequence
        product_indices = [self.product_to_idx.get(p, 0) for p in products]  # Use 0 for unknown products
        padded_products = self.pad_sequence(product_indices)
        
        if self.next_purchases is not None:
            next_purchases = self.next_purchases[idx]
            # Create multi-hot encoded vector for next purchases
            labels = torch.zeros(len(self.product_to_idx))
            for product in next_purchases:
                if product in self.product_to_idx:
                    labels[self.product_to_idx[product]] = 1
            return torch.tensor(padded_products, dtype=torch.long), torch.tensor(will_buy, dtype=torch.float), labels
        
        return torch.tensor(padded_products, dtype=torch.long), torch.tensor(will_buy, dtype=torch.float)


def create_data_loaders(train_df, valid_df, test_df, product_catalog, batch_size=32, max_seq_len=50):
    # Create product to index mapping
    product_to_idx = {p: i for i, p in enumerate(product_catalog)}
    
    # Create datasets with max_seq_len
    train_dataset = PurchaseDataset(train_df, product_to_idx, max_seq_len)
    valid_dataset = PurchaseDataset(valid_df, product_to_idx, max_seq_len)
    test_dataset = PurchaseDataset(test_df, product_to_idx, max_seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader, product_to_idx
