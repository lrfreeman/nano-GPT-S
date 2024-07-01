"""A simple bigram model that predicts the next character given the current character leveraging the PyTorch library.
This is based on the hero to zero series by Karpathy. NOTE to turn on PEDAGOGICAL_MODE to see logs explaining the code. 
Used as a teaching tool for myself to remeber how things work.

Positional information in a bigram model is not considered and therefore, the information is not used.

"""

from loguru import logger
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.embedding import Embedding
from utils.load_corpus import LoadCorpus
from utils.data_splitting import test_train_split, get_batch

class BigramLanguageModel(nn.Module):
    """A simple bigram model that predicts the next character given the current character.
    This is based on the hero to zero series by Karpathy."""

    def __init__(self, vocab_size, n_embed=32, block_size=8):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = self.create_lookup_table_for_embeddings(vocab_size)
        self.lm_head = nn.Linear(n_embed, vocab_size) # This is the output layer that maps the dense vector to the vocab size, language model head  
        self.position_embedding = nn.Embedding(block_size, n_embed)

    def create_lookup_table_for_embeddings(self, vocab_size, n_embed=32):
        """
        Creates a lookup table for embeddings using nn.Embedding.

        The nn.Embedding layer in PyTorch maps each input token to a dense vector of fixed size.
        In the context of a bigram model, we use a vocab_size by vocab_size matrix where each 
        token directly reads off the logits for the next token. At start the logits are randomly
        initialised and are updated during training.

        Args:
            vocab_size (int): The size of the vocabulary.
            n_embed (int): The number of dimensions to embed the characters in.

        Returns:
            nn.Embedding: The embedding layer initialized to vocab_size by vocab_size.
        """
        return nn.Embedding(vocab_size, n_embed)

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
        token_embed = self.token_embedding_table(idx) # (B,T,C) - C is the number of channels, i.e features in the embedding, this maps to n_embed in this example 
        pos_embed = self.position_embedding(torch.arange(T, device=idx.device)) # (T, C) Create a position embedding for each token in the sequence
        pos_embed = pos_embed.unsqueeze(0) # (1, T, C) Add a batch dimension to the position embedding
        # the C dimension is required so that each token can be mapped to a dense vector so that the model can learn the relationships between tokens
        x = token_embed + pos_embed # Add the token and position embeddings together
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
    
    # Hyper parameters
    test_split = 0.1
    block_size = 8 # Also known as the sequence length (LSTM) or context size (LLM or Transformer) - What is the maxium context size we want to consider?
    max_iters = 3000 # how many iterations to train for
    eval_interval = 100 # how often to evaluate the model
    lr = 1e-3 # learning rate
    eval_iters = 100 # how many iterations to average the loss over when evaluating
    n_embed = 32 # how many dimensions to embed the characters in
    batch_size = 4 # how many independent sequences will we process in parallel? 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": 
        logger.info("Using the GPU")  
    else: 
        logger.info("Using the CPU")
    #--------------------------------------------------

    torch.manual_seed(1337)
    
    logger.info("Running the bigram model")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    corpus = LoadCorpus(url).text
    embedding_obj = Embedding(corpus)
    data = embedding_obj.embedded_data
    train_data, val_data = test_train_split(data, test_split=test_split)
    model = BigramLanguageModel(vocab_size=embedding_obj.vocab_size)
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