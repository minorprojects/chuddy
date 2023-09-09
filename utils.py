import torch
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4
DATA_LINK= ''
SAVE_PATH = ''
