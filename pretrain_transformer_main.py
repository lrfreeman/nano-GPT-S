""" 

A simple bigram model that predicts the next character given the current character 
leveraging the PyTorch library. This is based on the hero to zero series by Karpathy. 

"""

from loguru import logger
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.embedding import Embedding
from utils.load_corpus import LoadCorpus
from utils.data_splitting import test_train_split, get_batch

# Hyper parameters --------------------------------
batch_size = 64 # how many independent sequences will we process in parallel?
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
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(in_features=n_embd, out_features=n_embd) # project the concatenated heads to the original dimension (n_embd
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    
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
    current token and thus a tril mask is used to prevent future tokens from influencing the current token."""
    
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
        wei = q @ k.transpose(-2, -1) # (B, T, T) # compute attention scores (afinity between keys and queries)
        wei = wei * C ** (-0.5) # scale by the dimensionality of the head size (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask out the future tokens (B, T, T)
        wei = F.softmax(wei, dim=-1) # normalise the weights (B, T, T)
        wei = self.dropout(wei) # apply dropout to the weights
        # perform communication between tokens
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head # why
        self.sa = MultiHeadAttention(n_head, head_size) # self attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) # what is this?
        
    def forward(self, x):
        # x is (B, T, Embd)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """ A simple character based model that utilises bigram statistics """

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embdding_table = nn.Embedding(vocab_size, n_embd) # This is the lookup table for the embeddings, maps each token to a dense vector
        self.position_embdding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)]) # This is the transformer block, it is a stack of blocks)
        self.ln_f = nn.LayerNorm(n_embd) # This is the final layer normalisation
        self.lm_head = nn.Linear(n_embd, vocab_size) # This is the output layer that maps the dense vector to the vocab size, language model head  

    def forward(self, idx, targets=None):
        """Using a set of fixed weights to predict the target token given the input sequence
        
        Args:
            -- idx (torch.Tensor | shape = (B, T)): 
                    The input sequence of tokens where B is the batch size and T is the sequence length.
            -- targets (torch.Tensor | shape = (B, T)): 
                    The target sequence of tokens of shape (B, T) where B is the batch size and T is the sequence 
                    length. NOTE: This is optional and only used during training, if empty the function will return 
                    None for loss and only produce logits."""
        
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
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
            # targets = targets.view(B*T) # Flatten the targets
            targets = targets.view(-1) # Flatten the targets
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
            # crop idx to the last block size
            idx_cond = idx[:, -block_size:] # (B, T) # Given we use positional embeddings we can have to crop to block size
            
            # get the predictions
            logits, _ = self(idx_cond) # loss is not needed as we are not training but generating

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

    #--------------------------------------------------


model = BigramLanguageModel()
model = model.to(device)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, block_size, batch_size, device=device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
optimiser = model.optimiser()

for iter in range(max_iters):
    
    # every once in a while, evaluate the loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        logger.info(f"Iteration: {iter}, Train loss: {losses['train']}, Val loss: {losses['val']}")
        
    # get a batch of data
    xb, yb = get_batch('train', train_data, val_data, block_size, batch_size, device=device) # Get a single bathc to see how the data looks like

    # evaluate the model
    logits, loss = model(xb, yb)
    model.optimiser().zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(embedding_obj.decode(model.generate(context, max_new_tokens=100)[0].tolist())) # Generate 100 new tokens
