# %% [markdown]
## Makemore part 4: becoming a backprop ninja
# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
# %%
# download the names.txt file from github
# !wget https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
# %%
# read in all the words
with open("names.txt", "r") as file:
    words = file.read().splitlines()
print(len(words))
print(max(len(w) for w in words))
print(words[:8])
# %%
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)
# %%
# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w:
            idx = stoi[ch]
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
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtrain, Ytrain = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])
# %%
# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f"{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}")
# %%
n_emd = 10 # the dimensionality of the character embedding vectors
n_hidden = 64 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
emb_matrix = torch.randn(vocab_size, n_emd, generator=g) # embedding matrix (27, 10)
# Layer 1
'''
W1 (30, 64)

multiple with 5/3 -> Xavier initialization 
5/3 factor is often associated with the initialization strategy for networks using the tanh activation function
it helps maintain the variance of activations across layers -> increase convergence

keep the activation and gradients in a reasonable range 
preventing them become too small (vanishing graidents) or too large (exploding graidents)

divided by (n_emd * block_size) ** 0.5 
(n_emd * block_size) is the number of input
(n_emd * block_size) ** 0.5 is the squre root of number of input 
why divide by it? scale the weights
'''
W1 = torch.randn((n_emd * block_size, n_hidden), generator=g) * (5/3) / ((n_emd * block_size)**0.5) 
b1 = torch.randn(n_hidden, generator=g)

# Layer 2
'''
W2 (64, 27)
'''
W2 = torch.randn((n_hidden, vocab_size), generator=g) 
b2 = torch.randn(vocab_size, generator=g)

# BatchNorm parameters
bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden)) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias] 
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True
# %%
batch_size = 32

# construct a minibatch
indices = torch.randint(0, Xtrain.shape[0], (batch_size, ), generator=g)
Xbatch, Ybatch = Xtrain[indices], Ytrain[indices] # batch X, Y

# %%
# Forward pass chunkated into smaller steps that are possible to backward one at a time

emb = emb_matrix[Xbatch] # embed the characters into vectors
embcat = emb.view(emb.shape[0], -1) # concatenate the vectors 
# linear layer 1
hprebn = embcat @ W1 + b1 # hidden layer pre-activation
# BatchNorm layer
bnmeani = 1/batch_size * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(batch_size-1) * (bndiff2).sum(0, keepdim=True)