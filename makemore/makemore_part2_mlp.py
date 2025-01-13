# %% [markdown]
## makemore part 2
# %% 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
# %matplotlib inline
# %%
words = open("names.txt", 'r').read().splitlines()
# %%
chars = sorted(list(set(''.join(words))))
ctoi = {s:i+1 for i, s in enumerate(chars)}
ctoi['.'] = 0
itoc = {i:s for s,i in ctoi.items()}
# %%
ctoi
# %%
block_size = 3 # how many characters do we take to predict the next one?
X, Y = [], []

for w in words[:10]:
    print(w)
    context = [0] * block_size
    for ch in w + ".":
        idx = ctoi[ch]
        print(f'index: {idx}')
        print(f'context: {[itoc[c] for c in context]} -> char: {ch}')
        X.append(context)
        Y.append(idx)
        context = context[1:] + [idx] # remove the first one and append

    print('')

X = torch.tensor(X)
Y = torch.tensor(Y)

# %%
X
# %%
Y.shape
Y
# %%
'''
why use embeddings?
[0,0,5] ..e
discrete representations -> one-hot encoding for these indices
we would end up with high-dimensional vectors (e.g., a vector of length 27 for each character)

in one-hot encoding, each unique character or token is represented as a vector where only one element is 1 (indicating the presence of that character) and the rest are 0

Embedding Transformation
1. embedding matrix (27, 2): 27 unique characters and 10 embedding dimension
each row corresponds to a character and contains 10-dimensional vector that represents that character in a continuous space
vector: array of number

2. mapping process:
when input an index (e.g., 5 for a character), use this index to select the corresponding row from the embedding matrix (this row is the character's embedding)
- maps the high-dimensional one-hot vector to a lower dimensional cintinuous vector

3. learning embeddings:
during training, the embedding matrix is updated through backpropagtion. the model learns to adjust the embeddings such that characters iwth similar roles or contexts in the data have similar embeddings
- allows embeddings to capture semantic relationships between characters

Advatnage using Embeddings:
- dimensionality reduction
- dense and continuous representation
- semantic meaning

Example:
One-Hot Vector: [0, 0, 0, 0, 0, 1, 0, ..., 0] (27-dimensional)
Embedding Vector: [0.12, -0.56, ..., 0.34] (10-dimensional)
'''
# 27 unique characters & 2 embedding dimensions
embedding_matrix = torch.randn((27,2)) 
embedding_matrix 
# %%
'''
Embedding Matrix (27,2):
 tensor([[ 0.1234, -0.5678],
         [ 0.2345,  0.6789],
         [ 0.3456, -0.7890],
         ...
         [ 0.4567,  0.8901],
         [ 0.5678, -0.9012]])

X indices (32, 3): 32 sequences, each with 3 indices
 tensor([[ 0,  0,  0],
         [ 0,  0,  5],
         [ 0,  5, 13],
         [ 5, 13, 13]])

Indexing: each index in X is used to select the corresponding row from the embedding matrix 
and map them into the result
0 -> [ 0.1234, -0.5678]
5 -> [ 0.6789,  0.1234]
Indexing for each sequence (32, 3, 2):
 tensor([[[ 0.1234, -0.5678],
          [ 0.1234, -0.5678],
          [ 0.1234, -0.5678]],

         [[ 0.1234, -0.5678],
          [ 0.1234, -0.5678],
          [ 0.6789,  0.1234]],

         [[ 0.1234, -0.5678],
          [ 0.6789,  0.1234],
          [-0.2345,  0.3456]],

         [[ 0.6789,  0.1234],
          [-0.2345,  0.3456],
          [-0.2345,  0.3456]]])

'''
# indexing
embedding_matrix_indexing = embedding_matrix[X]
print(f'X.shape {X.shape}')
print(f'embedding_matrix.shape {embedding_matrix.shape}')
print(f'embeddings.shape {embedding_matrix_indexing.shape}')
# %%
'''
Embedding Matrix (27, 2):
 tensor([[ 0.1234, -0.5678],
         [ 0.2345,  0.6789],
         [ 0.3456, -0.7890],
         [ 0.4567,  0.8901],
         # ... more rows ...
         ])

X Indices (32, 3): 32 sequences each with 3 indices
 tensor([[ 0,  0,  0], 
         [ 0,  0,  5],
         [ 0,  5, 13],
         [ 5, 13, 13]])

Indexing: each index in X is used to select the corresponding row from the embedding matrix 
0 -> [ 0.1234, -0.5678]
5 -> [ 0.6789,  0.1234]
Indexing for each sequence (32, 3, 2):
 tensor([[[ 0.1234, -0.5678],
          [ 0.1234, -0.5678],
          [ 0.1234, -0.5678]],

         [[ 0.1234, -0.5678],
          [ 0.1234, -0.5678],
          [ 0.6789,  0.1234]],

         [[ 0.1234, -0.5678],
          [ 0.6789,  0.1234],
          [-0.2345,  0.3456]],

         [[ 0.6789,  0.1234],
          [-0.2345,  0.3456],
          [-0.2345,  0.3456]]])

Embedding refer to transformation -> high-dimensional to lower-dimensional
as below example: (32,3,2) -> (32,6)

Embeddings (32, 6): 32 sequences each with 6 indices
0 -> [ 0.1234, -0.5678]
[0,0,0] -> [[ 0.1234, -0.5678], [ 0.1234, -0.5678], [ 0.1234, -0.5678]]
 tensor([[ 0.1234, -0.5678,  0.1234, -0.5678,  0.1234, -0.5678],
         [ 0.1234, -0.5678,  0.1234, -0.5678,  0.6789,  0.1234],
         [ 0.1234, -0.5678,  0.6789,  0.1234,  1.4567,  0.9012],
         [ 0.6789,  0.1234,  1.4567,  0.9012,  1.4567,  0.9012]])

''' 
embeddings = embedding_matrix_indexing.view(embedding_matrix_indexing.shape[0], 6) # (32, 6)
# %%
W1 = torch.randn((6,100))
b1 = torch.randn(100)
# %%
# torch.cat([emb[:, 0, :], emb[:,1,:], emb[:,2,:]], 1).shape
# torch.cat(torch.unbind(embeddings, 1), 1).shape
# embeddings.view(32, 6) == torch.cat(torch.unbind(embeddings, 1), 1)
# embeddings.storage() 

# %%
hidden_layer = torch.tanh(embeddings.view(embeddings.shape[0], 6) @ W1 + b1) # (32, 100)
hidden_layer
# %%
hidden_layer.shape # (32,100)
# %%
W2 = torch.randn((100, 27))
b2 = torch.randn(27)
# %%
logits = hidden_layer @ W2 + b2 
# %%
logits.shape
# %%
# softmax
counts = logits.exp() 
counts
# %%
prob = counts / counts.sum(1, keepdim=True)
prob
# %% 
prob.shape
# %%
# loss
loss = prob[torch.arange(32), Y].log().mean()
loss
# %%
X.shape, Y.shape # X.shape: (32,3) Y.shape: (32)
# %%
g = torch.Generator().manual_seed(2147483647)
embedding_matrix = torch.rand((27,2), generator=g)
W1 = torch.randn(6, 100, generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn(100, 27, generator=g)
b2 = torch.randn(27, generator=g)
parameters = [embedding_matrix, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
# %%
sum(p.nelement() for p in parameters)
# %%
'''
Embedding Matrix (27,2):
 tensor([[ 0.1234, -0.5678],
         [ 0.2345,  0.6789],
         [ 0.3456, -0.7890],
         ...
         [ 0.4567,  0.8901],
         [ 0.5678, -0.9012]])

X indices (32, 3): 32 sequences, each with 3 indices
 tensor([[ 0,  0,  0],
         [ 0,  0,  5],
         [ 0,  5, 13],
         [ 5, 13, 13]])

Indexing: each index in X is used to select the corresponding row from the embedding matrix 
and map them into the result
0 -> [ 0.1234, -0.5678]
5 -> [ 0.6789,  0.1234]

Embeddings refer to transformation that maps high-dimensional data into dense & lower-dimensional data
as below example: 
matrix (32,3,2) -> (32,6)

Indexing for each sequence (32, 3, 2):
 tensor([[[ 0.1234, -0.5678],
          [ 0.1234, -0.5678],
          [ 0.1234, -0.5678]],

         [[ 0.1234, -0.5678],
          [ 0.1234, -0.5678],
          [ 0.6789,  0.1234]],

         [[ 0.1234, -0.5678],
          [ 0.6789,  0.1234],
          [-0.2345,  0.3456]],

         [[ 0.6789,  0.1234],
          [-0.2345,  0.3456],
          [-0.2345,  0.3456]]])

Embeddings (32, 6): 32 sequences each with 6 indices
0 -> [ 0.1234, -0.5678]
[0,0,0] -> [[ 0.1234, -0.5678], [ 0.1234, -0.5678], [ 0.1234, -0.5678]]
 tensor([[ 0.1234, -0.5678,  0.1234, -0.5678,  0.1234, -0.5678],
         [ 0.1234, -0.5678,  0.1234, -0.5678,  0.6789,  0.1234],
         [ 0.1234, -0.5678,  0.6789,  0.1234,  1.4567,  0.9012],
         [ 0.6789,  0.1234,  1.4567,  0.9012,  1.4567,  0.9012]])

(32, 6) * (6, 100) + (100) -> (32, 100)
(32, 100) * (100, 27) + (27) -> (32, 27)
'''

learning_rate_powers = torch.linspace(-3, 0, 1000)
learning_rates = 10 ** learning_rate_powers

track_learning_rate = []
track_loss = []

for i in range(1000):
    # minibatch construct
    idx = torch.randint(0, X.shape[0], (32,))

    # Forward pass
    embedding_matrix_indexing = embedding_matrix[X[idx]] # (32, 3, 2) 
    embeddings = embedding_matrix_indexing.view(embedding_matrix_indexing.shape[0], 6)
    hidden_layer = torch.tanh(embeddings @ W1 + b1) # (32, 100)
    logits = hidden_layer @ W2 + b2 # (32, 27)
    # counts = logits.exp()
    # probs = counts / counts.sum(dim=1, keepdim=True)
    # loss = -prob[torch.arange(32), Y].log().mean()
    # ------- cross entropy -------
    loss = F.cross_entropy(logits, Y[idx])
    # print(loss.item())

    # Backward pass
    for p in parameters:
        p.grad = None

    loss.backward()

    # Update
    learning_rate = learning_rates[i]
    for p in parameters:
        p.data += -learning_rate * p.grad

    # track stats
    track_learning_rate.append(learning_rate_powers[i])
    track_loss.append(loss.item())
# %%
'''
find out the good learning rate is around 0.1
'''
plt.plot(track_learning_rate, track_loss)
# %%
'''
use the good learning rate 0.1 to train
'''
for _ in range(10000):
    # get random 32 indices from input -> indices[]
    indices = torch.randint(0, X.shape[0], (32,))

    # forward pass
    embedding_matrix_indexing = embedding_matrix[X[indices]] # (32,3,2)
    # flatten (32,3,2) to (32,6)
    embeddings = embedding_matrix_indexing.view(embedding_matrix_indexing.shape[0], 6)
    # W1: (6,100); b1 (100)
    # hidden layer: (32, 100)
    hidden_layer = torch.tanh(embeddings @ W1 + b1) 
    # W2: (100, 27); b2 (27)
    # output layer: (32, 27)
    logits = hidden_layer @ W2 + b2
    # Y[indices] -> 32 output 
    loss = F.cross_entropy(logits, Y[indices])

    # backward pass
    for p in parameters:
        p.grad = None
    
    loss.backward()

    # update
    learning_rate = 0.1
    for p in parameters:
        p.data += -learning_rate * p.grad

print(loss.item())
# %% [markdown]
## Let's split the data into training, dev, test set
# %%
# training split, dev/validation split, test split
# 80%, 10%, 10%

# build the dataset
def build_dataset(words):
    block_size = 3 # context length: how many characters do we take to predict the next one?
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            idx = ctoi[ch]
            X.append(context)
            Y.append(idx)
            context = context[1:] + [idx]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

import random
random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2])
# %%
g = torch.Generator().manual_seed(2147483647)
embedding_matrix = torch.randn((27,2))
W1 = torch.randn((6,300), generator=g)
b1 = torch.randn(300, generator=g)
W2 = torch.randn((300, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [embedding_matrix, W1, b1, W2, b2]
# %%
sum(p.nelement() for p in parameters)
# %%
for p in parameters:
    p.requires_grad = True
# %%
Xtr.shape, Ytr.shape
# %%
track_steps = []
track_loss = []

# %%
for i in range(10000):

    # minibatch
    # get random 32 indices from input -> indices[]
    indices = torch.randint(0, Xtr.shape[0], (32,))

    # fordward pass
    embedding_matrix_indexing = embedding_matrix[Xtr[indices]]
    embeddings = embedding_matrix_indexing.view(embedding_matrix_indexing.shape[0], 6)
    hidden_layer = torch.tanh(embeddings @ W1 + b1)
    logits = hidden_layer @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[indices])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    learning_rate = 0.1
    for p in parameters:
        p.data += -learning_rate * p.grad
    
    # track stats
    track_steps.append(i)
    track_loss.append(loss.item())
# %%
plt.plot(track_steps, track_loss)
# %%
'''
Next question: overfitting & underfitting
https://chatgpt.com/c/67793978-2c4c-8004-b3d2-c83f34e4a4f9
'''
