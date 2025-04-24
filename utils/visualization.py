import matplotlib.pyplot as plt
from typing import Dict

def plot_training_history(history: Dict, model_name: str = ''):
    plt.figure(figsize=(15, 10))
    
    metrics = list(history['train'].keys())
    num_metrics = len(metrics)
    
    for i, metric in enumerate(metrics):
        plt.subplot((num_metrics + 1) // 2, 2, i + 1)
        plt.plot(history['train'][metric], label=f'Train {metric}')
        plt.plot(history['val'][metric], label=f'Validation {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} over epochs')
        plt.legend()
    
    plt.suptitle(f'Training Metrics for {model_name}')
    plt.tight_layout()
    plt.show()