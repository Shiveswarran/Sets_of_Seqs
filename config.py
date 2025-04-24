import torch

class Config:
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    EMBEDDING_DIM = 512
    HIDDEN_DIM = 256
    DROPOUT = 0.3
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WILL_BUY_EPOCHS = 5
    RECOMMENDER_EPOCHS = 10
    
    # Prediction parameters
    TOP_K = 5
    
    # Data paths
    DATA_DIR = 'data'
    TRAIN_FILE = 'train.csv'
    VALID_FILE = 'valid.csv'
    TEST_FILE = 'test.csv'