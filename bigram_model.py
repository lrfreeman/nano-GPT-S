""" 

A simple decoder bigram model that predicts the next character given the current character 
leveraging the PyTorch library. This is based on the hero to zero series by Karpathy. And
is effectively a zero layer neural network (no hidden layers) that uses a lookup table (technically 
an embedding layer) to predict the next token given the current token.

"""

from loguru import logger
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.embedding import Embedding
from utils.load_corpus import LoadCorpus
from utils.data_splitting import test_train_split, get_batch

# Hyper parameters --------------------------------
max_new_tokens = 1000 # how many new tokens to generate
batch_size = 64 # 64 how many independent sequences will we process in parallel?
block_size = 256 # how many characters to process at once also kwown as the sequence length (RNNS) or context size (LLM or Transformer)
max_iters = 5000 # how many iterations to train for
eval_interval = 200 # how often to evaluate the model
eval_iters = 200 # how many iterations to average the loss over when evaluating
lr = 3e-4 # learning rate - bring it downn if model is bigger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logger.info("Using the GPU")
else:
    logger.info("Using the CPU")
n_embd = 64 # how many dimensions to embed the characters in
test_split = 0.1 # what percentage of the data to use for testing
#--------------------------------------------------

torch.manual_seed(1337)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
corpus = LoadCorpus(url).text
embedding_obj = Embedding(corpus)
data = embedding_obj.embedded_data
train_data, val_data = test_train_split(data, test_split=test_split)
vocab_size = embedding_obj.vocab_size

class BigramLanguageModel(nn.Module):
    """ A simple character based model that utilises bigram statistics """

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # This is the lookup table for the embeddings, maps each token to a dense vector

    def forward(self, idx, targets=None):
        """ Using a set of fixed weights to predict the target token given the input sequence
        
        Args:
            -- idx (torch.Tensor | shape = (B, T)): 
                    The input sequence of tokens where B is the batch size and T is the sequence length.
            -- targets (torch.Tensor | shape = (B, T)): 
                    The target sequence of tokens of shape (B, T) where B is the batch size and T is the sequence 
                    length. NOTE: This is optional and only used during training, if empty the function will return 
                    None for loss and only produce logits. """
        
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) - C is the number of channels, i.e features in the embedding, this maps to n_embd in this example 

        # If no targets are provided, return only the logits, this is so we generate text
        if targets is None:
            loss = None

        # If targets are provided, calculate the cross entropy loss
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # Flatten the logits
            assert logits.shape == (B*T, C), "The shape of the logits is not as expected."
            targets = targets.view(-1) # Flatten the targets
            loss = F.cross_entropy(logits, targets) # Compute the cross entropy loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Generates new tokens from the model given a context.
        
        Given a context of shape (B, T) where B is the batch size and T is the sequence length. The purpose of this
        function is to generate new tokens for each of the sequences in the batch up until max_new_tokens. So first
        we will predict (B, T + 1), and then (B, T + 2) and so on until we reach max_new_tokens.
        
        How does the self(idx) work?
            Every class has a special dunnder method called __call__ which is called when the class is called like a function.
            In pytorch, this is used to define the forward pass of the model. So when we call self(idx), we are calling the forward function above.
            
        Args:
            -- idx (torch.Tensor | shape = (B, T)): 
                    The input sequence of tokens where B is the batch size and T is the sequence length.
            -- max_new_tokens (int): 
                    The maximum number of tokens to generate for each sequence in the batch. """

        # idx is (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            # crop idx to the last block size
            logits, _ = self(idx) # get predictions - loss is not needed as we are not training but generating - (B, T, C)
            logits = logits[:, -1, :] # focus only on last time step which is what we use for prediction - (B, C)
            probs = F.softmax(logits, dim=-1) #  apply softmax to get probabilities (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) - For each sequence in the batch, predict the next token
            # append sampled index to the running sequence
            # 1 dimension is the time dimension, so given the current token, we predict the next token
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + i)
        return idx

    def optimiser(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


Model = BigramLanguageModel()
Model = Model.to(device)
optimiser = Model.optimiser()

@torch.no_grad() # Decorator to ensure that the function does not compute gradients for memory efficiency
def estimate_loss() -> dict:
    """ Averages the loss over many batches to check model performance 
    
    Returns:
        -- out (dict): 
                A dictionary containing the training and validation loss. """
    out = {}
    Model.eval() # Set the model to evaluation mode to ensure we don't update the weights
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, block_size, batch_size, device=device) # Uses train or val data depending on the split
            _, loss = Model(X, Y) # don't need the logits, just the loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    Model.train() # Set the model back to training mode in which we can update the weights
    return out

# ------------------------ TRAINING LOOP ------------------------

for iter in range(max_iters):
    
    # every once in a while, evaluate the loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        logger.info(f"Iteration: {iter}, Train loss: {losses['train']}, Val loss: {losses['val']}")
        
    # get a batch of data
    xb, yb = get_batch('train', train_data, val_data, block_size, batch_size, device=device) # Get a single bathc to see how the data looks like

    # evaluate the model
    _, loss = Model(xb, yb) # Don't need the logits, just the loss
    Model.optimiser().zero_grad(set_to_none=True) # Zero the gradients
    loss.backward() # Backpropagate the loss
    optimiser.step() # Update the weights

# ------------------------ GENERATE TEXT ------------------------

context = torch.zeros((1,1), dtype=torch.long, device=device) # Here zero refers to a new line character from our vocab
print(embedding_obj.decode(Model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())) # Generate 100 new tokens, the [0] is because generate works on batches, and we only have one sequence here
