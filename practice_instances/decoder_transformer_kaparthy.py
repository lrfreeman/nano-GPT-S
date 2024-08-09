""" 

Transformer decoder only architecture matching closely to attention is all you need papper leveraging the PyTorch library. 
This is based on the hero to zero series by Karpathy. 

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
eval_interval = 500 # how often to evaluate the model
lr = 3e-4 # learning rate - bring it downn if model is bigger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logger.info("Using the GPU")
else:
    logger.info("Using the CPU")
eval_iters = 200 # how many iterations to average the loss over when evaluating
n_embd = 384 # how many dimensions to embed the characters in
n_head = 6 # How many heads within the multihead attention framework
n_layer = 6
dropout = 0.2 # what percentage of neurons to drop out during training on each pass - Add when concerned about overfitting
test_split = 0.1 # what percentage of the data to use for testing
#--------------------------------------------------

torch.manual_seed(1337)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
corpus = LoadCorpus(url).text
embedding_obj = Embedding(corpus)
data = embedding_obj.embedded_data
train_data, val_data = test_train_split(data, test_split=test_split)
vocab_size = embedding_obj.vocab_size

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # Stacks a bunch of Head objects together in a list comprehension, each index is a head that can be accessed like a normal list
        self.proj = nn.Linear(in_features=num_heads * head_size, out_features=n_embd) # A projection layer that maps the concatenated heads back to the original embedding size
        self.dropout = nn.Dropout(dropout) # Regularisation technique to prevent overfitting
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1) # A way to pull all of the heads together and concatenate them along the last dimension (channel dimension)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple multi-layerd-perceptron (MLP) with each lyaer followed by a ReLU non-linearity]
        the ffn is to allow each token to have a more complex relationship with the communication it receives from other tokens
        in the sequence. This happens on a per token basis. I.e this is the computation part of the transformer block. 
        Where as the communication part is the self attention part. 
        
        In the og paper the embedding size is 512 and the hidden layer size is 2048. So we 
        leave here x4 as the hidden layer size."""
    
    # why *4 ?
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    """ One head of a self-attention decoder block.
    
    Decoders are used to predict the next token in a sequence given the
    current token and thus a tril mask is used to prevent future tokens 
    from influencing the current token. """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # What do I contain?
        self.query = nn.Linear(n_embd, head_size, bias=False) # What am I looking for?
        self.value = nn.Linear(n_embd, head_size, bias=False) # What should I communicate?
        # Buffers in pytorch are persistent and will not be updated by the optimizer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Lower triangular matrix
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        k = k.transpose(-2, -1) # (B, C, T) - Transpose the key tensor
        # (B, T, C) @ (B, C, T) = (B, T, T) - This is the dot product between the query and the key
        wei = q @ k # (B, T, T) # compute attention scores (afinity between keys and queries)
        wei = wei * C ** (-0.5) # scale by the dimensionality of the head size (B, T, T) - Stability trick 
        # to prevent the weights from getting too large or too small which can cause one hot encodings to dominate the softmax
        # Note that in wei the affinities are autoregressive with increasing time steps starting from the current token
        # and ending with predicting the last token in the sequence using all the previous tokens in the sequence
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask out the future tokens (B, T, T)
        wei = F.softmax(wei, dim=-1) # normalise the weights (B, T, T)
        wei = self.dropout(wei) # apply dropout to the weights
        # perform communication between tokens
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation. This is the decoder block
    that can be seen in the original transformer paper by Vaswani et al and stacked together to 
    form the transformer model.
    
    Note. In the original attentional is all you need paper had layer norm before the self attention. 
    Karpathy's implementation has it after as it now more common to do so """
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head # The size of each head is determined by the number of heads and the embedding size to maintain the dimensionality as if there was only one head
        self.sa = MultiHeadAttention(n_head, head_size) # self attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # Basically a fancy z-score normalisation with affine learnable parameters so your data doesn't need to be gaussian
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        """Where both the attention layer and the ffwd layer write to the residual stream"""
        # x is (B, T, Embd)
        x = x + self.sa(self.ln1(x)) # Self attention, layer normalisation and residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
    """ A simplified transformer model from the original paper "Attention is all you need" by Vaswani et al.
    and the hero to zero series by Karpathy. This model is a decoder only model and thus does not have an encoder. """

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embdding_table = nn.Embedding(vocab_size, n_embd) # This is the lookup table for the embeddings, maps each token to a dense vector
        self.position_embdding = nn.Embedding(block_size, n_embd) # Each position in the sequence also gets it's own embedding
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)]) # This is the transformer block, it is a stack of blocks)
        self.ln_f = nn.LayerNorm(n_embd) # This is the final layer normalisation
        self.lm_head = nn.Linear(n_embd, vocab_size) # This is the output layer that maps the dense vector to the vocab size, language model head  

    def forward(self, idx, targets=None):
        """ Using a set of fixed weights to predict the target token given the input sequence
        
        Args:
            -- idx (torch.Tensor | shape = (B, T)): 
                    The input sequence of tokens where B is the batch size and T is the sequence length.
            -- targets (torch.Tensor | shape = (B, T)): 
                    The target sequence of tokens of shape (B, T) where B is the batch size and T is the sequence 
                    length. NOTE: This is optional and only used during training, if empty the function will return 
                    None for loss and only produce logits. """
        
        B, T = idx.shape # idx and targets are both (B,T) tensor of integers
        token_embd = self.token_embdding_table(idx) # (B,T,C) - C is the number of channels, i.e features in the embedding, this maps to n_embd in this example 
        pos_embed = self.position_embdding(torch.arange(T, device=idx.device)) # (T, C) Create a position embedding for each token in the sequence
        pos_embed = pos_embed.unsqueeze(0) # (1, T, C) Add a batch dimension to the position embedding
        # the C dimension is required so that each token can be mapped to a dense vector so that the model can learn the relationships between tokens
        x = token_embd + pos_embed # Add the token and position embeddings together
        x = self.blocks(x) # (B, T, C) - Apply the self attention head
        x = self.ln_f(x) # (B, T, C) - Apply the layer normalisation
        logits = self.lm_head(x) # (B,T, vocab_size)

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
            idx_cond = idx[:, -block_size:] # (B, T) # Given we use positional embeddings we can have to crop to block size
            logits, _ = self(idx_cond) # get predictions - loss is not needed as we are not training but generating - (B, T, C)
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

    #--------------------------------------------------


Model = Transformer()
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
