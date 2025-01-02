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
block_size = 3 # how many characters do we take to predict the next one?
X, Y = [], []

for w in words:
    # print(w)
    context = [0] * block_size
    for ch in w + ".":
        idx = ctoi[ch]
        # print(f'index: {idx}, char: {ch}, context: {context}')
        X.append(context)
        Y.append(idx)
        # print(''.join(itoc[i] for i in context), '----->', itoc[idx])
        context = context[1:] + [idx] # remove the first one and append

    # print('')

X = torch.tensor(X)
Y = torch.tensor(Y)

# %%
X
# %%
Y.shape
Y
# %%
'''
(vocab_size, embedding_dim)
vocab_size is the number of unique items -> num_classes
embedding_dim is the size of each embedding vector
'''
embedding_matrix = torch.randn((27,2))
embedding_matrix # (27 items each with 2 dimensional embedding)
# %%
one_row_tensor = F.one_hot(torch.tensor(5), num_classes=27).float()
one_row_tensor
one_row_tensor @ embedding_matrix
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
Embeddings for each sequence (32, 3, 2):
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
embeddings = embedding_matrix[X]
print(f'X.shape {X.shape}')
print(f'embedding_matrix.shape {embedding_matrix.shape}')
print(f'embeddings.shape {embeddings.shape}')
# %%
'''
Flatten the embeddings for each sequence (32, 6)

Embedding Matrix (27, 2):
 tensor([[ 0.1234, -0.5678],
         [ 0.2345,  0.6789],
         [ 0.3456, -0.7890],
         [ 0.4567,  0.8901],
         # ... more rows ...
         ])

Item Indices (32, 3): 32 sequences each with 3 indices
 tensor([[ 0,  0,  0], 
         [ 0,  0,  5],
         [ 0,  5, 13],
         [ 5, 13, 13]])

Flattened Embeddings (32, 6): 32 sequences each with 6 indices
0 -> [ 0.1234, -0.5678]
[0,0,0] -> [[ 0.1234, -0.5678], [ 0.1234, -0.5678], [ 0.1234, -0.5678]]
 tensor([[ 0.1234, -0.5678,  0.1234, -0.5678,  0.1234, -0.5678],
         [ 0.1234, -0.5678,  0.1234, -0.5678,  0.6789,  0.1234],
         [ 0.1234, -0.5678,  0.6789,  0.1234,  1.4567,  0.9012],
         [ 0.6789,  0.1234,  1.4567,  0.9012,  1.4567,  0.9012]])

''' 
flatten_embeddings = embeddings.view(embeddings.shape[0], 6) # (32, 6)
# %%
W1 = torch.randn((6,100))
b1 = torch.randn(100)
# %%
# torch.cat([emb[:, 0, :], emb[:,1,:], emb[:,2,:]], 1).shape
# torch.cat(torch.unbind(embeddings, 1), 1).shape
embeddings.view(32, 6) == torch.cat(torch.unbind(embeddings, 1), 1)
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
# %% [markdown]
'''
------------ now made respectable :)    ------------
'''
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
Embeddings for each sequence (32, 3, 2):
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

Flattened Embeddings (32, 6): 32 sequences each with 6 indices
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
    embeddings = embedding_matrix[X[idx]] # (32, 3, 2) 
    hidden_layer = torch.tanh(embeddings.view(embeddings.shape[0], 6) @ W1 + b1) # (32, 100)
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
    embeddings = embedding_matrix[X[indices]] # (32,3,2)
    # flatten (32,3,2) to (32,6)
    flatten_embeddings = embeddings.view(embeddings.shape[0], 6)
    # W1: (6,100); b1 (100)
    # hidden layer: (32, 100)
    hidden_layer = torch.tanh(flatten_embeddings @ W1 + b1) 
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
# %%
