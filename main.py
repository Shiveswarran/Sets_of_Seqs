import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List,Dict, Tuple
import ast
from collections import defaultdict
import matplotlib.pyplot as plt
from config import Config
from models.encoder import ProductEmbedder, PurchaseSequenceEncoder
from models.classifier import WillBuyClassifier
from models.recommender import ProductRecommender
from utils.dataloader import PurchaseDataset, create_data_loaders
from utils.metrics import MetricsTracker
from utils.visualization import plot_training_history

def load_data(data_dir, train_file, valid_file, test_file):
    train_df = pd.read_csv(os.path.join(data_dir, train_file))
    valid_df = pd.read_csv(os.path.join(data_dir, valid_file))
    test_df = pd.read_csv(os.path.join(data_dir, test_file))
    return train_df, valid_df, test_df

def get_product_catalog(dfs: list) -> List[str]:
    all_products = set()
    for df in dfs:
        # Extract products from products_before
        products_before = df['products_before'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        for products in products_before:
            all_products.update(products)
        
        # Extract products from post_cutoff if available
        if 'post_cutoff' in df.columns:
            post_cutoff = df['post_cutoff'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            for products in post_cutoff:
                all_products.update(products)
    
    return list(all_products)

def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    epochs,
    task_type='binary'
):
    history = {'train': defaultdict(list), 'val': defaultdict(list)}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_metrics = MetricsTracker(task_type)
        
        for batch in train_loader:
            # Check if batch has 2 or 3 elements
            if len(batch) == 2:
                products, labels = batch
                will_buy = None
            else:
                products, will_buy, labels = batch
            
            will_buy = will_buy.to(device).float().view(-1, 1)
            
            # Convert products to embeddings
            # (This would be replaced with actual encoding)
            inputs = torch.randn(len(products), Config.EMBEDDING_DIM).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, will_buy)
            loss.backward()
            optimizer.step()
            
            train_metrics.update(loss.item(), outputs, will_buy)
        
        # Validation phase
        model.eval()
        val_metrics = MetricsTracker(task_type)
        
        with torch.no_grad():
            for batch in valid_loader:
                # Check if batch has 2 or 3 elements
                if len(batch) == 2:
                    products, labels = batch
                    will_buy = None
                else:
                    products, will_buy, labels = batch
                
                will_buy = will_buy.to(device).float().view(-1, 1)
                
                inputs = torch.randn(len(products), Config.EMBEDDING_DIM).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, will_buy)
                val_metrics.update(loss.item(), outputs, will_buy)
        
        # Update history
        train_metrics_dict = train_metrics.get_metrics()
        val_metrics_dict = val_metrics.get_metrics()
        
        for metric in train_metrics_dict:
            history['train'][metric].append(train_metrics_dict[metric])
            history['val'][metric].append(val_metrics_dict[metric])
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_metrics_dict["loss"]:.4f} | '
              f'Acc: {train_metrics_dict["accuracy"]:.4f} | '
              f'Prec: {train_metrics_dict.get("precision", 0):.4f} | '
              f'Rec: {train_metrics_dict.get("recall", 0):.4f} | '
              f'F1: {train_metrics_dict.get("f1", 0):.4f}')
        print(f'Val Loss: {val_metrics_dict["loss"]:.4f} | '
              f'Acc: {val_metrics_dict["accuracy"]:.4f} | '
              f'Prec: {val_metrics_dict.get("precision", 0):.4f} | '
              f'Rec: {val_metrics_dict.get("recall", 0):.4f} | '
              f'F1: {val_metrics_dict.get("f1", 0):.4f}')
        print('-' * 80)
    
    return history


def main():
    # Load data
    train_df, valid_df, test_df = load_data(
        Config.DATA_DIR, Config.TRAIN_FILE, Config.VALID_FILE, Config.TEST_FILE
    )
    
    # Get product catalog
    product_catalog = get_product_catalog([train_df, valid_df, test_df])
    
    # Create data loaders
    train_loader, valid_loader, test_loader, product_to_idx = create_data_loaders(
        train_df, valid_df, test_df, product_catalog, Config.BATCH_SIZE
    )
    
    # Initialize models
    embedder = ProductEmbedder()
    encoder = PurchaseSequenceEncoder(embedder)
    will_buy_model = WillBuyClassifier().to(Config.DEVICE)
    recommender_model = ProductRecommender(Config.EMBEDDING_DIM, len(product_catalog)).to(Config.DEVICE)
    
    # Train will-buy classifier
    print("Training Will-Buy Classifier...")
    will_buy_criterion = torch.nn.BCELoss()
    will_buy_optimizer = optim.Adam(will_buy_model.parameters(), lr=Config.LEARNING_RATE)
    
    will_buy_history = train_model(
        will_buy_model,
        train_loader,
        valid_loader,
        will_buy_criterion,
        will_buy_optimizer,
        Config.DEVICE,
        Config.WILL_BUY_EPOCHS,
        'binary'
    )
    plot_training_history(will_buy_history, "Will-Buy Classifier")
    
    # Train product recommender
    print("\nTraining Product Recommender...")
    recommender_criterion = torch.nn.BCELoss()
    recommender_optimizer = optim.Adam(recommender_model.parameters(), lr=Config.LEARNING_RATE)
    
    recommender_history = train_model(
        recommender_model,
        train_loader,
        valid_loader,
        recommender_criterion,
        recommender_optimizer,
        Config.DEVICE,
        Config.RECOMMENDER_EPOCHS,
        'multilabel'
    )
    plot_training_history(recommender_history, "Product Recommender")
    
    # Evaluate on test set
    # (Implementation would be similar to validation)
    
    # Save models
    torch.save(will_buy_model.state_dict(), 'will_buy_model.pth')
    torch.save(recommender_model.state_dict(), 'recommender_model.pth')

if __name__ == "__main__":
    main()