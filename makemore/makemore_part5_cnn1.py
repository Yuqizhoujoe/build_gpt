# %% [markdown]
## makemore part5
'''
- video: https://www.youtube.com/watch?v=t3YJ5hKiMQ0&ab_channel=AndrejKarpathy
- github: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb
- makemore github: https://github.com/karpathy/makemore
- colab: https://colab.research.google.com/drive/1CXVEmCO_7r7WYZGb5qnjfyxTvQa13g5X?usp=sharing
'''
# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
# %%
# read in all the words
words = open("names.txt", 'r').read().splitlines()
print(len(words))
print(max(len(w) for w in words))
print(words[:8])
# %%
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)
# %%
# shuffle up the words
import random 
random.seed(42)
random.shuffle(words)
# %%
# build the dataset
block_size = 8 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
    X, Y = [], []
    
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            idx = stoi[ch]
            X.append(context)
            Y.append(idx)
            context = context[1:] + [idx] 
            
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtrain, Ytrain = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1: n2])
Xtest, Ytest = build_dataset(words[n2:])
# %%
for x,y in zip(Xtrain[:20], Ytrain[:20]):
    print(''.join(itos[idx.item()] for idx in x), '-->', itos[y.item()])
# %%
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn(fan_in, fan_out) / fan_in**0.5 # kaiming init
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
      
# ---------------------------------------------------------------------------
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            '''
            e = torch.randn(32,4,68)
            # emean = e.mean(0, keepdim=True) # (1,4,68)
            emean = e.mean((0,1), keepdim=True) # (1,1,68)
            # evar = e.var(0, keepdim=True) # (1,4,68)
            evar = e.var((0,1), keepdim=True) # (1,1,68)
            ehat = (e - emean) / torch.sqrt(evar + 1e-5) # (32,4,68)
            ehat.shape
            '''
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
# ---------------------------------------------------------------------------
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

# ---------------------------------------------------------------------------
class Embedding:
    def __init__(self, num_embeddings, embedding_dim) -> None:
        self.weight = torch.randn(num_embeddings, embedding_dim)
    
    def __call__(self, index):
        self.out = self.weight[index]
        return self.out

    def parameters(self):
        return [self.weight]

# ---------------------------------------------------------------------------
# %%
e = torch.randn(4,8,10) # goal: want this to be (4,4,20) where consecutive 10D vectos get concatenated
'''
e[:, ::2, :] selects every second element along the second dimension starting from the first element -> reduce second dimension from 8 to 4 (taking elements at indices 0, 2, 4, and 6)

e[:, 1::2, :] selects every second element along the second dimension starting from the second element -> reduces the second dimension from 8 to 4 taking elements at indices 1,3,5, and 7

torch.cat([...], dim=2) concatenates the 2 tensors alonmg the last dimension 
each of the selected slices from e has a shape of (4,4,10) -> (4,4,20)
'''
explicit = torch.cat([e[:, ::2, :], e[:, 1::2, :]], dim=2)
explicit.shape
# %%
e.view(4,4,20) == explicit
# %%
'''
The process of transform a tensor from (B,8,10) to (B,80) through intermdiate steps like (B,4,20) is structured way to progressively combine features from consecutive elements

Flattening: (B, 8, 10) to (B, 4, 20)
Processing: Apply a linear layer or activation function
Flattening: (B, 4, 20) to (B, 2, 40)
Processing: Another transformation
Final Flattening: (B, 2, 40) to (B, 80)
This structured approach allows the model to learn and refine features at each stage, potentially leading to better performance compared to a direct flattening.
'''
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
        
    '''
    B: batch size
    T: Sequence length or time steps, which is the number of elements in a sequence.
    C: Number of channels or features, which is the number of features per element in the sequence.
    '''
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self):
        return []
    
# ---------------------------------------------------------------------------
class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
# %%
torch.manual_seed(42)
# %%
n_embd = 24 # the dimensionality of the character embedding vectors
n_hidden = 128 # the number of neurons in the hidden layer of the MLP
model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1 # last layer make less confident

parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True
# %%
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    # minibatch construct
    indices = torch.randint(0,  Xtrain.shape[0], (batch_size, ))
    Xbatch, Ybatch = Xtrain[indices], Ytrain[indices] 
    
    # forward pass
    logits = model(Xbatch)
    loss = F.cross_entropy(logits, Ybatch) 
    
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # update: simple SGD
    lr = 0.1 if i < 150000 else 0.01 # step learning decay
    for p in parameters:
        p.data += -lr * p.grad
    
    # track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())
# %%
# len(lossi) 200000
# torch.tensor(lossi).view(-1, 1000) (200, 1000)
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1)) 
# %%
for layer in model.layers:
    print(layer.__class__.__name__, ":", tuple(layer.out.shape))
    layer.training = False
# %%
# evaluate the loss
@torch.no_grad() # this decorator disables gradient tracking inside pytorch
def split_loss(split):
    x,y = { 
        'train': (Xtrain, Ytrain),
        'val': (Xdev, Ydev),
        'test': (Xtest, Ytest)
    }[split]
    
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())
    
split_loss('train')
split_loss('val')

# %% [markdown]
### performance log
'''
- original (3 character context + 200 hidden neurons, 12K params): train 2.058, val 2.105
- context: 3 -> 8 (22K params): train 1.918, val 2.027
- flat -> hierarchical (22K params): train 1.941, val 2.029
- fix bug in batchnorm: train 1.912, val 2.022
- scale up the network: n_embd 24, n_hidden 128 (76K params): train 1.769, val 1.993
'''
# %%
# sample from the model

for _ in range(20):
    # Initialize an empty list to store the generated indices
    out = []
    # Start with a context of zeros, which is the initial input to the model
    context = [0] * block_size 
    
    while True:
        # Forward pass the neural net with the current context
        logits = model(torch.tensor([context]))
        
        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=1)
        
        # Sample an index from the probability distribution
        index = torch.multinomial(probs, num_samples=1).item()
        
        # Shift the context window and track the samples
        context = context[1:] + [index]
        
        # Append the sampled index to the output list
        out.append(index)
        
        # If we sample the special '.' token, break the loop
        if index == 0:
            break
    
    # Decode the generated indices into characters and print the generated word
    generated_word = ''.join(itos[i] for i in out)
# %%
for x,y in zip(Xtrain[7:15], Ytrain[7:15]):
    print(''.join(itos[idx.item()] for idx in x), '-->', itos[y.item()])
# %%
# forward a single example:
logits = model(Xtrain[[7]])
logits.shape
# %%
# forward all of them
logits = torch.zeros(8, 27)
for i in range(8):
  logits[i] = model(Xtrain[[7+i]])
  print(logits[i])
  print(Xtrain[[7+i]])
logits.shape
# %%
