import pandas as pd
import numpy as np
import torch
import ast
from sentence_transformers import SentenceTransformer
from dataset import BasketDataset
from torch.utils.data import DataLoader
from model import TextModels
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

def accuracy_score(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == labels).float().mean()
    return acc.item(), preds


def make_permuted_loader(dataset, batch_size=64, device="cpu"):
    def collate_permute(batch):
        baskets, labels = BasketDataset.collate(batch)
        baskets_perm = [t[torch.randperm(len(t))].to(device) for t in baskets]
        labels = labels.to(device)
        return baskets_perm, labels
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=False,   
                      collate_fn=collate_permute)



def main():


    train_csv = "data/train.csv"
    val_csv   = "data/valid.csv"
    test_csv  = "data/test.csv"
    all_csvs  = [train_csv, val_csv, test_csv]

    train_ds = BasketDataset(train_csv, all_csvs)
    val_ds   = BasketDataset(val_csv,   all_csvs)
    test_ds  = BasketDataset(test_csv,  all_csvs)


    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                            collate_fn=BasketDataset.collate)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False,
                            collate_fn=BasketDataset.collate)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False,
                            collate_fn=BasketDataset.collate)
    test_loader_permuted = make_permuted_loader(test_ds, batch_size=64, device="cpu")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Model = TextModels().to(device)
    print(f"\n**************************Model**************************\n{Model}")


    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, Model.parameters()),
        lr=1e-4,
        weight_decay = 1e-4
    )

    n_epochs = 100
    train_losses = []
    train_accs = []
    val_accs = []
    val_losses = []

    best_val_loss = float('inf')
    patience = 5      
    epochs_no_improve = 0
    best_state_dict = None


    for epoch in range(1, n_epochs + 1):

        print(f"\n****************************Epoch {epoch} of {n_epochs}****************************\n")

        Model.train()
        total_loss = 0
        total_acc = 0


        for baskets, labels in train_loader:
            baskets = [b.to(device) for b in baskets]
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = Model(baskets)
            loss = Model.loss_fn(logits, labels)
            acc, _ = accuracy_score(logits, labels)
            # loss = Model.loss(baskets, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        Model.eval()
        all_logits, all_labels = [], []
        val_loss = 0

        with torch.no_grad():
            for baskets, labels in val_loader:
                baskets = [b.to(device) for b in baskets]
                labels = labels.to(device)
                logits = Model(baskets)
                all_logits.append(logits)
                all_labels.append(labels)
                loss = Model.loss_fn(logits, labels)
                val_loss += loss.item()

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        val_acc, _ = accuracy_score(logits, labels)
        avg_val_loss = val_loss / len(val_loader)

        val_accs.append(val_acc)
        val_losses.append(avg_val_loss)

        # print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = {k: v.cpu() for k, v in Model.state_dict().items()}
            epochs_no_improve = 0
            print("  ↳ new best val-loss … saving weights")
        else:
            epochs_no_improve += 1
            print(f"  ↳ no improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs!")
            break


    # epochs_range = range(1, n_epochs + 1)
    epochs_range = range(1, len(train_losses) + 1)
    if best_state_dict is not None:
        Model.load_state_dict(best_state_dict)


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Training Accuracy')
    plt.plot(epochs_range, val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    # plt.show() # This might not work in non-GUI environments
    f_out_dim = Model.model_out_shape
    plot_filename = f"training_plots_{n_epochs}_{f_out_dim}.png"
    plt.savefig(plot_filename, dpi = 600)
    print(f"Plots saved to {plot_filename}")


    print("\nEvaluating on the test set...\n")
    Model.eval()
    all_test_logits, all_test_labels = [], []

    with torch.no_grad():
        for baskets, labels in test_loader:
            baskets = [b.to(device) for b in baskets]
            labels = labels.to(device)
            logits = Model(baskets)
            all_test_logits.append(logits)
            all_test_labels.append(labels)

    test_logits = torch.cat(all_test_logits)
    test_labels = torch.cat(all_test_labels)
    test_acc, _ = accuracy_score(test_logits, test_labels)

    print(f"Test Accuracy: {test_acc:.4f}")




    print("\n--- METRICS---")

    def evaluate_and_print_metrics(loader, dataset_name):
        print(f"\nEvaluating on {dataset_name} set...")
        Model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for baskets, labels in loader:
                baskets = [b.to(device) for b in baskets]
                labels = labels.to(device)
                logits = Model(baskets)
                all_logits.append(logits)
                all_labels.append(labels)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        acc, preds = accuracy_score(logits, labels)

        # probs = torch.sigmoid(logits)
        # preds = (probs >= 0.5).float()

        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print(f"{dataset_name} Set Metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        cm_filename = f"confusion_matrix_{dataset_name.lower()}_{n_epochs}_{f_out_dim}.png"
        # np.savetxt(cm_filename, cm, delimiter=",", fmt='%d')
        # print(f"Confusion matrix saved to {cm_filename}")

        plt.figure(figsize=(6,5))
        sns.heatmap(cm,
                    annot=True,   
                    fmt='d',           
                    cmap='plasma',      
                    cbar=True,         
                    linewidths=1,       
                    linecolor='white')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        plt.tight_layout()
        plt.savefig(cm_filename, dpi=600)

    # Evaluate on Training Set (requires a non-shuffled loader)
    train_loader_eval = DataLoader(train_ds, batch_size=64, shuffle=False, # Use shuffle=False for consistent evaluation
                                   collate_fn=BasketDataset.collate)
    evaluate_and_print_metrics(train_loader_eval, "Training")

    # Evaluate on OriginalTest Set
    evaluate_and_print_metrics(test_loader, "Test")

    # Evaluate on Permuted Test Set
    evaluate_and_print_metrics(test_loader_permuted, "Permuted_Test")


if __name__ == "__main__":
    main()

