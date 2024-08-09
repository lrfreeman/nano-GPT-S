"""A basic implementation of a transformer model. Building on Karpathy's lecture series and including knowledge from the Arena 3 course.
I prefer this implementation using einops as the parameters are more explicit and the code is more readable. I also like the use of dataclasses
and type hints to make the code more readable and maintainable. I have also included a training loop that uses the wandb library for logging. Notably
the einops implementation skips having a seperate Head class and a Multi-head class which is common in ohter implementations"""

from loguru import logger
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import einops
from jaxtyping import Float, Int
import wandb
from typing import Tuple, List, Optional, Dict, Callable
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import datasets
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformer_lens import HookedTransformer

from utils.embedding import Embedding
from utils.load_corpus import LoadCorpus
from utils.data_splitting import test_train_split, get_batch
from configs import ModelConfig, TransformerTrainingArgs

class PosEmbed(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.W_pos = nn.Parameter(torch.empty(cfg.n_ctx, cfg.d_model))
        nn.init.normal_(self.W_pos, std=cfg.init_range)
    
    def forward(self, token: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        batch_size, seq_len = token.shape
        return einops.repeat(self.W_pos[:seq_len], 'T d_model -> B T d_model', B=batch_size)

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return einops.einsum(normalized_resid_final, self.W_U, "batch posn d_model, d_model d_vocab -> batch posn d_vocab") + self.b_U

class Attention(nn.Module):
    """A multi-headed attention layer that allows each token to attend to other tokens in the sequence."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("tril", torch.tensor(-1e5, dtype=torch.float32, device=device))

    def apply_causal_mask(self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """Applies a causal mask to attention scores, and returns masked scores."""
        all_ones = torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device) # Create a matrix of all ones
        mask = torch.triu(all_ones, diagonal=1).bool() # Create a mask that is True above the diagonal
        attn_scores.masked_fill_(mask, self.tril) # Mask the scores
        return attn_scores

    def forward(self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        # Sum over d_model as that dimensions dissapears in the einsum
        Q = einops.einsum(normalized_resid_pre, self.W_Q, "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head") + self.b_Q # (B, T, nheads, d_head)
        K = einops.einsum(normalized_resid_pre, self.W_K, "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head") + self.b_K # (B, T, nheads, d_head)
        V = einops.einsum(normalized_resid_pre, self.W_V, "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head") + self.b_V # (B, T, nheads, d_head)
        # Calculate attention scores
        attention = einops.einsum(Q, K, "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K") / torch.sqrt(torch.tensor(self.cfg.d_head)) # (B, nheads, T, T)
        mask = self.apply_causal_mask(attention) # (B, nheads, T, T)
        attention = torch.softmax(mask, dim=-1) # Apply softmax to the last dimension which is the key position (B, nheads, pos_Q, pos_K)
        # Apply attention to values
        Z = einops.einsum(V, attention, "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head") # (B, T, nheads, d_head)
        results = einops.einsum(Z, self.W_O, "batch posn_Q nheads d_head, nheads d_head d_model  -> batch posn_Q nheads d_model") 
        # sum over the nheads dimension and add bias
        output = einops.einsum(results, "batch posn_Q nheads d_model -> batch posn_Q d_model") + self.b_O
        return output

class MLP(nn.Module):
    """A simple multi-layerd-perceptron (MLP) with each lyaer followed by a ReLU non-linearity
    the ffn is to allow each token to have a more complex relationship with the communication it receives from other tokens
    in the sequence. This happens on a per token basis."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.ReLU(),
            nn.Linear(cfg.d_mlp, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ A single transformer block that contains a self attention layer and a feed forward layer. """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        
    def forward(self, residual_pre: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_model"]:
        normalised_residual_pre = self.ln1(residual_pre)
        resid_mid = residual_pre + self.attn(normalised_residual_pre)
        normalised_resid_mid = self.ln2(resid_mid)
        resid_out = resid_mid + self.mlp(normalised_resid_mid)
        return resid_out

class Transformer(nn.Module):
    """A basic instance of a transformer model with a decoder only architecture for an NLP task. The model
    inputs a sequence of tokens and computes a probability distribution over the corpus and has a simple 
    generate function for sampling new tokens from the model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.positional_embedding = PosEmbed(cfg)
        self.transformer_blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_blocks)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = Unembed(cfg)
    
    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        """Ingests a batched sequence of tokens and turns them into a set of logits"""
        token_embd = self.token_embedding(tokens) # (B, T, d_model)
        pos_embd = self.positional_embedding(tokens) # (B, T)
        residual_stream = token_embd + pos_embd # (B, T, d_model)
        for block in self.transformer_blocks:
            residual_stream = block(residual_stream)
        unembed = self.unembed(residual_stream) # (B, T, d_vocab)
        return unembed
    
    def generate(self):
        raise NotImplementedError("This function is not yet implemented")
    
class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: Transformer):
        super().__init__()
        self.model = model
        self.args = args
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0

    def training_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        '''
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        '''
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        loss = -get_log_probs(logits, tokens).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"training_loss": loss}, step=self.step)
        return loss


    def validation_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]):
        '''
        Calculates & returns the accuracy on the tokens in the batch (i.e. how often the model's prediction
        is correct). Logging should happen in the `train` function (after we've computed the accuracy for 
        the whole validation set).
        '''
        tokens = batch["tokens"].to(device)
        logits: Tensor = self.model(tokens)[:, :-1] # Aligning predictions with targets by removing the shift
        predictions = logits.argmax(dim=-1) # Get the index of the highest logit
        correct_predictions = (predictions == tokens[:, 1:]) # Compare the predictions with the actual tokens
        return correct_predictions # Return the accuracy


    def train(self):
        '''
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        '''
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = np.nan
        
        progress_bar = tqdm(total=self.args.epochs * self.args.max_steps_per_epoch * self.args.epochs)
        
        for epoch in tqdm(range(self.args.epochs)):
            for i, batch in enumerate(self.train_loader()):
                loss = self.training_step(batch)
                description = f"Epoch {epoch+1}/{self.args.epochs}, loss: {loss.item():.4f}"
                progress_bar.set_description(description)
                if i >= self.args.max_steps_per_epoch:
                    break
                
            # Calculate validation accuracy
            correct_predictions = torch.concatenate([self.validation_step(batch) for batch in self.test_loader()])
            accuracy = correct_predictions.float().mean().item()
            wandb.log({"validation_accuracy": accuracy}, step=self.step)
        wandb.finish()
            
    def train_loader(self) -> DataLoader:
        '''Returns train loader (as in code above).'''
        return DataLoader(dataset_dict["train"], batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


    def test_loader(self) -> DataLoader:
        '''Returns test loader (as in code above).'''
        return DataLoader(dataset_dict["test"], batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
def get_log_probs(logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return log_probs_for_tokens

    #--------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": logger.info("Using the GPU")
    else: logger.info("Using the CPU")
    
    # configs
    mcfg = ModelConfig()
    tcfg = TransformerTrainingArgs()
    
    reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device
    )
    
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
    tokenized_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=mcfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
    dataset_dict = tokenized_dataset.train_test_split(test_size=1000) 
    train_loader = DataLoader(dataset_dict["train"], batch_size=tcfg.batch_size, shuffle=True, num_workers=4, pin_memory=True) 
    test_loader = DataLoader(dataset_dict["test"], batch_size=tcfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ------------------------ Run model --------------------------
    mcfg.d_vocab = reference_gpt2.cfg.d_vocab
    model = Transformer(mcfg).to(device)
    trainer = TransformerTrainer(tcfg, model)
    trainer.train()
    
