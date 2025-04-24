import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict
import torch

class MetricsTracker:
    def __init__(self, task_type: str = 'binary'):
        self.task_type = task_type
        self.reset()
        
    def reset(self):
        self.loss = 0.0
        self.correct = 0
        self.total = 0
        self.all_preds = []
        self.all_labels = []
        
    def update(self, loss: float, outputs: torch.Tensor, labels: torch.Tensor):
        batch_size = labels.size(0)
        self.loss += loss * batch_size
        
        if self.task_type == 'binary':
            predicted = (outputs > 0.5).float()
            self.correct += (predicted == labels).sum().item()
        else:
            predicted = (outputs > 0.5).float()
            self.correct += (predicted == labels).sum().item() / labels.shape[1]
        
        self.total += batch_size
        self.all_preds.append(outputs.detach().cpu().numpy())
        self.all_labels.append(labels.detach().cpu().numpy())
        
    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        metrics['loss'] = self.loss / self.total
        metrics['accuracy'] = self.correct / self.total
        
        if len(self.all_preds) > 0:
            preds = np.concatenate(self.all_preds)
            labels = np.concatenate(self.all_labels)
            
            if self.task_type == 'binary':
                preds = (preds > 0.5).astype(int)
                metrics['precision'] = precision_score(labels, preds, zero_division=0)
                metrics['recall'] = recall_score(labels, preds, zero_division=0)
                metrics['f1'] = f1_score(labels, preds, zero_division=0)
            else:
                preds = (preds > 0.5).astype(int)
                metrics['precision'] = precision_score(labels, preds, average='micro', zero_division=0)
                metrics['recall'] = recall_score(labels, preds, average='micro', zero_division=0)
                metrics['f1'] = f1_score(labels, preds, average='micro', zero_division=0)
                
        return metrics