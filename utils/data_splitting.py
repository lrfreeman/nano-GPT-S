from loguru import logger
import torch

def test_train_split(data, test_split=0.1):
    """Split the data into train and validation sets"""
    n = int((1-test_split)*len(data)) # first 1-test_split% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(split, train_data, val_data, block_size, batch_size, device, PEDAGOGICAL_MODE=False):
    """Extract random sequences of length block_size from the data. Return as input and target pairs.
    NOTE - The batches are not sequential and may overlap.
    
    Overlapping pros and cons:
    -- Pros:
        - More data to learn from
        - Good for small datasets
    -- Cons:
        - could lead to overfitting
        - computationally expensive
        
    """
    # generate a small batch of data of inputs x and targets y
    if split == 'train':
        data = train_data
    else:
        data = val_data
    
    # Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
    # If batch size is 4, then we will get 4 random integers between 0 and len(data) - block_size
    ix = torch.randint(low = 0, high = len(data) - block_size, size = (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix], dim = 0)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix], dim = 0)
    
    if PEDAGOGICAL_MODE:
        logger.info("The batches are random and may overlap. This is good for small datasets but could lead to overfitting. It is computationally expensive but provides more data to learn from.")
        logger.info("Be mindful on the correct indexing of the input and target sequences.")
    
    x, y = x.to(device), y.to(device)
    return x, y