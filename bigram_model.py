"""A simple bigram model that predicts the next character given the current character leveraging the PyTorch library.
This is based on the hero to zero series by Karpathy. NOTE to turn on PEDAGOGICAL_MODE to see logs explaining the code. 
Used as a teaching tool for myself to remeber how things work."""

from loguru import logger
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.embedding import Embedding
from utils.load_corpus import LoadCorpus
from utils.context_explorer import explore_how_context_works
from utils.data_splitting import test_train_split, get_batch

class BigramLanguageModel(nn.Module):
    """A simple bigram model that predicts the next character given the current character.
    This is based on the hero to zero series by Karpathy."""

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = self.create_lookup_table_for_embeddings(vocab_size)

    def create_lookup_table_for_embeddings(self, vocab_size):
        """
        Creates a lookup table for embeddings using nn.Embedding.

        The nn.Embedding layer in PyTorch maps each input token to a dense vector of fixed size.
        In the context of a bigram model, we use a vocab_size by vocab_size matrix where each 
        token directly reads off the logits for the next token. At start the logits are randomly
        initialised and are updated during training.
        
        NOTE:
         -- Keen to dig deeper into why not applying softmax here to convert logits to probabilities.

        Args:
            vocab_size (int): The size of the vocabulary.

        Returns:
            nn.Embedding: The embedding layer initialized to vocab_size by vocab_size.
        """
        return nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Using a set of fixed weights to predict the target token given the input sequence
        
        Args:
            -- idx (torch.Tensor | shape = (B, T)): 
                    The input sequence of tokens where B is the batch size and T is the sequence length.
            -- targets (torch.Tensor | shape = (B, T)): 
                    The target sequence of tokens of shape (B, T) where B is the batch size and T is the sequence 
                    length. NOTE: This is optional and only used during training, if empty the function will return 
                    None for loss and only produce logits."""

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) - C is the number of channels, in this case the vocab size

        # If no targets are provided, return only the logits, this is so we generate text
        if targets is None:
            loss = None

        # If targets are provided, calculate the cross entropy loss
        else:
            B, T, C = logits.shape
            logits_reshaped = torch.einsum('btc->(bt)c', logits)
            logits = logits_reshaped
            assert logits.shape == (B*T, C), "The shape of the logits is not as expected."
            # targets = targets.view(B*T) # Flatten the targets
            targets = torch.einsum('bt->(bt)', targets) # Flatten the target into a 1d tensor
            loss = F.cross_entropy(logits, targets) # Compute the cross entropy loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generates new tokens from the model given a context.
        
        NOTE - How do we generate in batches? We can't generate in batches as the model is autoregressive surely?
        
        How does the self(idx) work?
            Every class has a special dunnder method called __call__ which is called when the class is called like a function.
            In pytorch, this is used to define the forward pass of the model. So when we call self(idx), we are calling the forward function above."""

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx) # loss is not needed as we are not training but generating

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) - For each sequence in the batch, predict the next token

            # append sampled index to the running sequence
            # 1 dimension is the time dimension, so given the current token, we predict the next token
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def optimiser(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == "__main__":
    
    PEDAGOGICAL_MODE = True
    logger.info("Running the bigram model")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    corpus = LoadCorpus(url).text
    embedding_obj = Embedding(corpus, PEDAGOGICAL_MODE=PEDAGOGICAL_MODE)
    data = embedding_obj.embedded_data
    
    # Hyper parameters
    torch.manual_seed(1337)
    test_split = 0.1
    block_size = 8 # Also known as the sequence length (LSTM) or context size (LLM or Transformer)
    batch_size = 4 # how many independent sequences will we process in parallel?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": 
        logger.info("Using the GPU")  
    else: 
        logger.info("Using the CPU")


    # Split the data into train and validation sets
    train_data, val_data = test_train_split(data, test_split=test_split)
    
    if PEDAGOGICAL_MODE:
        explore_how_context_works()
        xb, yb = get_batch('train', train_data, val_data, block_size, batch_size, device=device, P) # Get a single bathc to see how the data looks like
        logger.info(f"Input batch shape: {xb.shape}, Target batch shape: {yb.shape}")
        print(f"Input batch (in embedded form): \n {xb}")
        logger.info("Note that the number of rows or examples in the batch is the batch size. The number of columns or features is the block size. The target batch is the input batch shifted by one.")
        model = BigramLanguageModel(vocab_size=embedding_obj.vocab_size)
        model = model.to(device)
        logits, loss = model(xb, yb)
        logger.info(f"Note the vocab size is: {embedding_obj.vocab_size}. This is the number of unique characters in the text.")
        logger.info(f"The logits have shape: {logits.shape}. This is the (batch size * block size, vocab size) tensor.")
        logger.info("As there is one prediction of the next token per sequence you have this first dimension. The second dimension is the logits applied to each token in the vocabulary.")
        logger.info("Thus at each sequence you basically make 65 predictions (assuming the vocab size is 65) assigning a logit to each possible token.")
        logger.info(f"The way to think about the loss or cross entropy in this case is that we have a 1 / {embedding_obj.vocab_size} chance in predicting the next token correctly.")
        logger.info(f"The loss and the logits are: {loss}, {logits}")
    
    idx = torch.zeros((1, 1), dtype=torch.long, device=device) # This is the start token to kick off the generation. Single batch and single token
    print(embedding_obj.decode(model.generate(idx, max_new_tokens=100)[0].tolist()))
