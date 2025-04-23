class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve
    after certain patience.
    """
    def __init__(self, patience=3, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum difference between new loss and old loss for improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif (self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
import logging
import os
from datetime import datetime

def get_logger(log_dir="logs", log_name=None):
    """
    Returns a logger that logs messages to both the console and a log file.
    
    Args:
        log_dir (str): Directory where log file will be saved. Defaults to "logs".
        log_name (str): Log file name. If None, a name based on the current timestamp will be used.
    
    Returns:
        logger (logging.Logger): Configured logger object.
    """
    # Create the logging directory if it does not exist.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a log file name with a timestamp if not provided.
    if log_name is None:
        log_name = datetime.now().strftime("log_%Y%m%d_%H%M%S.log")
    log_file = os.path.join(log_dir, log_name)
    
    # Create logger and set its global log level.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Avoid adding duplicate handlers if the logger already has them.
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler: save log messages to the specified log file.
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler: also log messages to the console.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Define the log format.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add both handlers to the logger.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger