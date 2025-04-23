import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime

from helpers import EarlyStopping
from dataloader import PurchaseBinaryDataset, PurchaseMultiLabelDataset
from model import PurchaseBinaryClassifier, ProductMultiLabelPredictor
from helpers import get_logger
import argparse

def train_binary_classifier(
    model, train_loader, val_loader,
    device, lr, num_epochs, patience
):
    """
    Trains the PurchaseBinaryClassifier model on training data.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    early_stopper = EarlyStopping(patience=patience)
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for products_batch, label_batch in train_loader:
            products_list = list(products_batch)
            labels = label_batch.to(device).view(-1, 1)  # shape: (batch_size, 1)
            
            optimizer.zero_grad()
            logits = model(products_list)  
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for products_batch, label_batch in val_loader:
                products_list = list(products_batch)
                labels = label_batch.to(device).view(-1, 1)
                logits = model(products_list)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break
        
        
def evaluate_binary_classifier(model, data_loader, device):
    """
    Evaluate the binary classifier model on a given DataLoader.
    Returns average loss and accuracy.
    """
    criterion = nn.BCEWithLogitsLoss()
    model.eval().to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for products_batch, label_batch in data_loader:
            products_list = list(products_batch)
            labels = label_batch.to(device).view(-1, 1)
            logits = model(products_list)  # (B,1)
            loss = criterion(logits, labels.to(device).view(-1, 1))
            total_loss += loss.item()
            # Compute accuracy
            preds = torch.sigmoid(logits)
            predicted = (preds >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def train_multilabel_classifier(
    model, train_loader, val_loader,
    device, lr, num_epochs, patience
):
    """
    Trains the ProductMultiLabelPredictor model on training data
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    early_stopper = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for products_batch, label_batch in train_loader:
            # products: list of product lists
            products_list = list(products_batch) 
            labels = label_batch.to(device)  # shape: (batch_size, num_candidates)
            optimizer.zero_grad()
            logits = model(products_list)  # (B, num_candidates)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        # Average training loss  
        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for products_batch, label_batch in val_loader:
                products_list = list(products_batch)
                labels = label_batch.to(device)
                logits = model(products_list)
                val_losses.append(criterion(logits, labels).item())
        avg_val_loss = np.mean(val_losses)


        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break
        
def evaluate_multilabel_classifier(model, data_loader, device, threshold=0.5):
    """
    Evaluate the multi-label predictor model on a given DataLoader.
    Returns average loss and an example measure of "average precision at threshold."
    """
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    model.to(device)

    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for products_batch, label_batch in data_loader:
            products_list = list(products_batch)
            labels = label_batch.to(device)          
            
            logits = model(products_list)            
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    avg_loss = total_loss / len(data_loader)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    # For a simple metric, compute multi-label accuracy or exact match ratio at threshold, or F1, etc.
    # Here we'll do a simple example: average "precision" at threshold (not strictly MAP).
    preds_binary = (all_preds >= threshold).float()
    # True positives / predicted positives across entire dataset
    tp = (preds_binary * all_labels).sum().item()
    pred_positives = preds_binary.sum().item()
    precision = tp / pred_positives if pred_positives > 0 else 0.0

    return avg_loss, precision

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f">>> Using device: {device}")

logger = get_logger()  # Optionally, you can specify log_dir and log_name
logger.info("Starting the training process...")

df_train = pd.read_csv('/scratch/umni5/a/shives/Course_Projects/CS_587_DL/Project/10perc_h&m/train.csv')
df_valid = pd.read_csv('/scratch/umni5/a/shives/Course_Projects/CS_587_DL/Project/10perc_h&m/valid.csv')
df_test = pd.read_csv('/scratch/umni5/a/shives/Course_Projects/CS_587_DL/Project/10perc_h&m/test.csv')

df_train = df_train.sample(frac=0.1, random_state=42)
df_valid = df_valid.sample(frac=0.1, random_state=42)
df_test = df_test.sample(frac=0.1, random_state=42)


df_train_bin = df_train[['products_before', 'order_after']]
df_val_bin = df_valid[['products_before', 'order_after']]
df_test_bin = df_test[['products_before', 'order_after']]

df_train_multi = df_train[['products_before', 'post_cutoff']]
df_val_multi = df_valid[['products_before', 'post_cutoff']]
df_test_multi = df_test[['products_before', 'post_cutoff']]

all_products = (
    df_train['products_before'].explode().tolist() +
    df_train['post_cutoff'].explode().tolist() +
    df_valid['products_before'].explode().tolist() +
    df_valid['post_cutoff'].explode().tolist() +
    df_test['products_before'].explode().tolist() +
    df_test['post_cutoff'].explode().tolist()
)

candidate_products = list(set([item.strip() for sublist in all_products for item in eval(sublist) if sublist]))

#Prepare DataLoaders for Binary

train_dataset_bin = PurchaseBinaryDataset(df_train_bin)
val_dataset_bin   = PurchaseBinaryDataset(df_val_bin)
test_dataset_bin  = PurchaseBinaryDataset(df_test_bin)

train_loader_bin = DataLoader(train_dataset_bin, batch_size=args.batch_size, shuffle=True,pin_memory=True,num_workers=4)
val_loader_bin   = DataLoader(val_dataset_bin, batch_size=args.batch_size, shuffle=False,pin_memory=True,num_workers=2)
test_loader_bin  = DataLoader(test_dataset_bin, batch_size=args.batch_size, shuffle=False,pin_memory=True,num_workers=4)


# Train/Eval the Binary Classifier


binary_model = PurchaseBinaryClassifier(pretrained_model_name='t5-small', device=device).to(device)
train_binary_classifier(
    model=binary_model,
    train_loader=train_loader_bin,
    val_loader=val_loader_bin,
    device=device,
    lr=args.lr,
    num_epochs=args.epochs,
    patience=args.patience
)
test_loss_bin, test_acc_bin = evaluate_binary_classifier(binary_model, test_loader_bin, device=device)
print(f"\n[Binary] Test Loss: {test_loss_bin:.4f}, Test Accuracy: {test_acc_bin:.4f}")



# Prepare DataLoaders for Multi-Label

train_dataset_multi = PurchaseMultiLabelDataset(df_train_multi, candidate_products)
val_dataset_multi   = PurchaseMultiLabelDataset(df_val_multi, candidate_products)
test_dataset_multi  = PurchaseMultiLabelDataset(df_test_multi, candidate_products)

train_loader_multi = DataLoader(train_dataset_multi, batch_size=args.batch_size, shuffle=True)
val_loader_multi   = DataLoader(val_dataset_multi, batch_size=args.batch_size, shuffle=False)
test_loader_multi  = DataLoader(test_dataset_multi, batch_size=args.batch_size, shuffle=False)


#Train/Eval the Multi-Label Classifier

multi_model = ProductMultiLabelPredictor(candidate_products, pretrained_model_name='t5-small', device=device).to(device)
train_multilabel_classifier(
    model=multi_model,
    train_loader=train_loader_multi,
    val_loader=val_loader_multi,
    device=device,
    lr=args.lr,
    num_epochs=args.epochs,
    patience=args.patience
)
test_loss_multi, test_precision_multi = evaluate_multilabel_classifier(multi_model, test_loader_multi, device=device)
print(f"\n[Multi-Label] Test Loss: {test_loss_multi:.4f}, Test Precision: {test_precision_multi:.4f}\n")
