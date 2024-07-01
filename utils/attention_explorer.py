# Description: This script is used to explore the attention mechanism in the transformer model.
# Attention is reprsented by a weighted mean of the input sequence. In order for a given token to understand
# how important each token preeceding it is, it can do a weighted mean. We can use a trainable weight matrix
# to prevent future tokens from influencing the current token. This is done by masking.

"""Use can find the affinity of each token to each other by using matrix multipliciation and a lower triangular matrix."""

import torch
from torch.nn import functional as F
from torch import nn

# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print("Given a (3, 2) matrix b, we can compute a weighted average of the rows of b using a (3, 3) weight matrix a.")
print("Because the context length is 3 and we don't want weights impacting the future, we use a lower triangular matrix.")
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=a@b')
print(c)
print('------------------------------------')

# Because each row is a batch, and each column is a index in the sequence, the weight is the same.
# So in this case you can use this logic to do matrix multiplication to compute averages

# consider the following toy example:
torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
print(x.shape) # (4, 8, 2)

# We can manually calculate the weighted averages by loops but this is inefficient
# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C)) # bow is bag of words
for b in range(B): # for each batch
    for t in range(T): # for each time step
        xprev = x[b,:t+1] # (t,C) # calculate the mean of all the previous tokens in the sequence and store in xbow and then move to the next token
        xbow[b,t] = torch.mean(xprev, 0)

print(f"The matrix result when we loop through the data for the first batch is: {xbow[0]}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# version 2: using matrix multiply for a weighted aggregation - Set the weights to be fixed for now
wei = torch.tril(torch.ones(T, T)) # lower triangular matrix, this mask prevents future tokens from influencing the current token
wei = wei / wei.sum(1, keepdim=True) # normalise the weights
xbow2 = torch.einsum('tt,btc->btc', wei, x) # (T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)
print(wei)

# Of course, in practice, the weights are learned. Also, by just doing a sum. This process is very lossy. We have losed a ton of information about
# how token interact with each other and how they are positionally related. This is where the transformer model comes in. It uses forms of attention to solve this.

#---------------------------------------------------------------------------------------------------

# Weight each embedded token in the sequence by a fixed weight matrix, and sum them up
# The weight matrix says: What is the affinity of each token to each other token in the sequence? Do they influence each other and if so, they should communicate? 

# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
print("\n Here is version 3 mask before norm. ")
print(wei)
wei = F.softmax(wei, dim=-1)
xbow3 = torch.einsum('tt,btc->btc', wei, x) # (T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow3)

# ---------------------------------------------------------------------------------------------------

# Self attention example 
# Previously we have been making the weights uniform and fixed. However we need to learn those weights in a cleverl way
# Specifally, if i'm a vowel that always follows a consonant, then I should have a high weight to that consonant. This is what the model should learn.
# or if king is followed by queen, then king should have a high weight to queen. This is what the model should learn if word embeddings are used rather than character embeddings.

# 1. Every token at each position emits two vectors: a query vector, and a key vector
#-- The query vector is: "What am I looking for?"
#-- The key vector is: "what do I contain?"
# 2, To find the affinity between keys and querys, we take the dot product of the query and key vectors
# 3. this dot product becomes the new weight matrix - wei 

# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C) # embedded data shaped into batches, time steps and channels (features)

# let's see a single Head perform self-attention
head_size = 16
# learn a map from the channels to the head size
key = nn.Linear(C, head_size, bias=False) # What do I contain?
query = nn.Linear(C, head_size, bias=False) # What am I looking for?
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, H)
q = query(x) # (B, T, H)
# compute the dot product of the queries and keys to find the affinities between all tokens
k = k.transpose(-2, -1) # (B, H, T) # transpose (swap) the last two dimensions, -2 is the second last dimension and -1 is the last dimension
wei =  q @ k # (B, T, H) @ (B, H, T) ---> (B, T, T) # How much does each token care about each other token

print(f"In the self-attention dot product between the queries and keys, the raw output is (before we mask and normalise with the softmax): {wei[0]}")

tril = torch.tril(torch.ones(T, T)) # lower triangular matrix of ones, upper triangular matrix of zeros
wei = wei.masked_fill(tril == 0, float('-inf')) # mask out the upper triangular part of the weight matrix
wei = F.softmax(wei, dim=-1) # -inf will be zero after softmax because of the way softmax works

print("Printing the first batch of the weight matrix", wei[0])
print("here we can see that for the last row, the 4th and 7th token were pretty interesting to the last token")

# We then do another linear map from the key,query product to create a value space
# x is private information to that token, what position am I, what Im looking for and what do I contain. And we need to share this information
# v is the shared information between different tokens
v = value(x)
out = wei @ v # now aggregate the values based on the weights

print(out.shape)