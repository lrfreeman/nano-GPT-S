import torch
from loguru import logger

class Embedding:
    def __init__(self, text, PEDAGOGICAL_MODE=False):
        self.text = text
        self.vocab_size = len(set(text))
        self.chars = sorted(list(set(text)))
        self.char_to_ix, self.ix_to_char = self.embedding()
        self.embedded_data = self.encode(text)
        
        if PEDAGOGICAL_MODE:
            logger.info(f"The first 10 characters in the text are: {self.text[:10]}")
            logger.info(f"The first 10 integers in the embedded data are: {self.embedded_data[:10]}")
            logger.info("Note that the length of each sequence depends on how we embed the data. E.g if we do character-level embedding the sequence length matches the level of characters. If we do word-level embedding the sequence length matches the level of words. Ignoring the context size for now.")
        
    def embedding(self):
        """Convert characters to integers"""
        char_to_ix = {ch:i for i, ch in enumerate(self.chars)}
        ix_to_char = {i:ch for i, ch, in enumerate(self.chars)}
        return char_to_ix, ix_to_char
    
    def encode(self, string):
        """Convert characters to integers"""
        embedding = [self.char_to_ix[ch] for ch in string]
        torch_embedding = torch.tensor(embedding, dtype=torch.long)
        logger.info(f"Embedding shape and type: {torch_embedding.shape}, {torch_embedding.dtype}")
        return torch.tensor(embedding, dtype=torch.long)
    
    def decode(self, array):
        """Convert integers to characters"""
        return ''.join([self.ix_to_char[i] for i in array])
        