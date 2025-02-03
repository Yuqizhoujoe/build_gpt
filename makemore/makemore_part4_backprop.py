# %% [markdown]
## Makemore part 4: becoming a backprop ninja
'''
- build_makemore_backprop ninja.ipynb: https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing#scrollTo=8sFElPqq8PPp
- video: https://www.youtube.com/watch?v=q8SA3rM6ckI&ab_channel=AndrejKarpathy
- github: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb
'''
# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

parameters = [emb_matrix, W1, b1, W2, b2, bngain, bnbias] 
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
# --------------
'''
Batch Norm
1. Compute the Mean:
   - Calculate the mean of the batch for each feature
   - Formula: μ = (1/m) * Σ x_i
   - This centers the data around zero.

2. Compute the Variance:
   - Calculate the variance of the batch for each feature.
   - Formula: σ^2 = (1/m) * Σ (x_i - μ)^2
   - This measures the spread of the data.

3. Normalize:
   - Subtract the mean and divide by the standard deviation.
   - Formula: x̂_i = (x_i - μ) / sqrt(σ^2 + ε)
   - This scales the data to have a mean of 0 and a variance of 1.
   - ε is a small constant added for numerical stability.

4. Scale and Shift:
   - Apply a learned scale (γ) and shift (β) to the normalized data.
   - Formula: y_i = γ * x̂_i + β
   - This allows the network to undo the normalization if needed.

hprebn.sum(0, keepdim=True) sum all elements along the first dimension (rows) for each column
hprebn.shape (batch_size, n_hidden)
- hprebn.sum(0, keepdim=True).shape (1, n_hidden)
- hprebn.sum(0).shape (n_hidden, )

Example:
hprebn = torch.tensor([
   [1,2,3],
   [4,5,6],
   [7,8,9]
])
hprebn.sum(0) -> tensor([12,15,18])
hprebn.sum(0, keepdim=True) -> tensor([[12,15,18]])
'''
# compute the mean
bnmeani = 1/batch_size * hprebn.sum(0, keepdim=True) 

# compute the variance
# 1e-5 is ε the small constant for numerical stability -> avoid numerator to be 0
bndiff = hprebn - bnmeani 
bndiff2 = bndiff**2
bnvar = 1/(batch_size-1) * (bndiff2).sum(0, keepdim=True) # bessel's correction
bnvar_inv = (bnvar + 1e-5)**-0.5 # 1/sqrt(σ^2 + ε)

bnraw = bndiff * bnvar_inv # (x_i - μ) / sqrt(σ^2 + ε)

# scale and shift: y_i = γ * x̂_i + β
hpreact = bngain * bnraw + bnbias
# --------------

# non-linearity
# hidden layer
h = torch.tanh(hpreact) 

# linear layer 2
# --------------
'''
computes the raw scores (logits) for each class before applying any normalization like softmax
'''
logits = h @ W2 + b2 # output layer
# --------------

# Calculate cross entropy loss
# --------------
''' 
For numericall stability
the maximum value in each row of logits is subtracted from every element in that row
prevents large exponentials in the next step
'''
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # subtract max for numerial stability

# softmax
'''
softmax function converts the logits into probabilities for each class -> sum to 1
'''
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
# if use (1.0 / counts_sum) instead then can't get backprop to be bit exact
counts_sum_inv = counts_sum**-1 
'''
final probability distribution for each row
'''
probs = counts * counts_sum_inv

'''
cross entropy loss measures how well the predicted probabilities match the true labels
https://en.wikipedia.org/wiki/Cross-entropy#Estimation

1. Select the log probabilities of the true classes:
logprobs[range(batch_size), Ybatch] selectes the log probability of the true class for each sample in the batch. This is done by using the row index from range(batch_size ) and the column index from Ybatch 

2. Negate the log probabilities:
   - Cross-entropy loss is defined as the negative log likelihood of the true class.
   - Negating the log probabilities converts them into positive loss values.

3. Compute the mean:
   - Average the negative log probabilities across the batch to obtain a single scalar loss value.

Formula:
   - Cross-entropy loss for a single sample: L = -log(p(y_true))
   - For a batch of size N: Loss = -1/N * Σ log(p(y_i)), where p(y_i) is the predicted probability of the true class for the i-th sample.
'''
logprobs = probs.log() # logarithm of the probabilities
loss = -logprobs[range(batch_size), Ybatch].mean()
# --------------

# backward pass
for p in parameters:
    p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
          norm_logits, logit_maxes, logits, h, hpreact, bnraw, 
          bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
          embcat, emb]:
    t.retain_grad()
loss.backward()
loss
# %%
'''
Exercise 1: backprop through the whole thing manually
backpropagting through exactly all of the variables
as they are defined in the forward pass above, one by one

d -> derivative
'''

dlogprobs = torch.zeros_like(logprobs)

# loss = -1/3a + -1/3b + -1/3c
# dloss/da = -1/3 -> -1/n
# loss = -logprobs[range(batch_size), Ybatch].mean()
dlogprobs[range(batch_size), Ybatch] = -1.0/batch_size
# logprobs = probs.log()
dprobs = (1.0/probs) * dlogprobs
'''
probs = counts * counts_sum_inv
counts_sum_inv = 1 / counts_sum
counts_sum = counts.sum(1, keepdim=True).
   
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
Why Use sum(1)?
1. Reduction in Forward Pass:
   - In the forward pass, operations like summing elements in a row reduce a dimension.
2. Gradient Accumulation:
   - During backpropagation, gradients need to be accumulated across the same dimension
     that was reduced in the forward pass.
3. Broadcasting and Consistency:
   - Using sum(1, keepdim=True) ensures that the resulting tensor from the summation
     maintains the same number of dimensions as the input.
   - This is important for maintaining consistency in tensor shapes during gradient
     calculations, especially when broadcasting is involved.
'''
# counts_sum_inv = 1/counts_sum
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
dcounts = dprobs * counts_sum_inv 
'''
dcounts_sum_inv / dcounts_sum = -(1/counts_sum**2)
derivative of 1/x with respect ti x is -1/x**2
'''
# counts_sum_inv = 1/counts_sum
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
'''
counts_sum = counts.sum(1, keepdim=True)
counts = [a,b,c]; counts_sum = a + b + c
dcounts_sum / dcounts[i] = 1 
dloss / dcounts[i] -> dloss / dlogprobs * ... dcounts_sum / dcounts[i] -> dcounts_sum
'''
dcounts += torch.ones_like(counts) * dcounts_sum
'''
counts = norm_logits.exp()
the derivative of exp(x) with respect to x is exp(x) 
here is counts itself 
'''
dnorm_logits = dcounts * counts
'''
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes

regarding logit_maxes, in the forward pass max() is used to find the maximum value in each row of the logits tensor

in the backward pass, why sum(1)?
since logit_maxes is a single value subtracted from each element in the row, its gradient is the accumulation of the effect of this subtraction across all elements in the row
So we sum the gradients acrosss the row to reflect this cumulative effect

why keepdim=True?
ensures that the resulting tensor has the same number of dimensions as the input -> consistent tensor shapes 
'''
dlogits = dnorm_logits.clone() # copy of dnorm_logits
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
'''
foward pass: logit_maxes = logits.max(1, keepdim=True).values

Use of one_hot in Backpropagation:
- Max Operation: In the forward pass, logits.max(1) finds the max value in each row.
- Gradient Propagation: During backpropagation, gradients should only affect the max elements.
- One-hot Encoding: F.one_hot creates a mask where only max positions are 1.
- Purpose: Ensures dlogit_maxes is applied only to max elements in dlogits.
'''
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
'''
logits = h @ W2 + b2

- Forward Pass:
  - h: (batch_size, n_hidden)
  - W2: (n_hidden, vocab_size)
  - logits: (batch_size, vocab_size)

 - Gradient w.r.t. W2:
  - Transpose h to align dimensions: h^T @ dlogits
  - h^T: (n_hidden, batch_size)
  - dlogits: (batch_size, vocab_size)
  - Resulting shape matches W2: (n_hidden, vocab_size)

- Gradient w.r.t. h:
  - Transpose W2 to align dimensions: dlogits @ W2^T
  - dlogits: (batch_size, vocab_size)
  - W2^T: (vocab_size, n_hidden)
  - Resulting shape matches h: (batch_size, n_hidden)

- Gradient w.r.t. b2:
  - Sum gradients across batch: sum(dlogits, dim=0)
  - dlogits: (batch_size, vocab_size)
  - Resulting shape matches b2: (vocab_size)

- Purpose: Align dimensions for correct matrix multiplication during gradient computation.
'''
dW2 = h.T @ dlogits # (n_hidden, batch_size) @ (batch_size, vocab_size) -> (n_hidden, vocab_size)
'''
dim=0 refers to the rows of the tensor
when dim=0, you are summing down the columns -> collapse the rows into a single row

dim=1 refers to the columns of the tensor
when dim=1, you are summing across the rows -> collapse the columns into a single column

example:
[[a, b, c],
 [d, e, f],
 [g, h, i]]
 
[a+d+g, b+e+h, c+f+i]  # Resulting in a single row

[a+b+c, d+e+f, g+h+i]  # Resulting in a single column
'''
db2 = dlogits.sum(dim=0) # Sum across batch dimension -> (vocab_size) 
dh = dlogits @ W2.T # (batch_size, vocab_size) @ (vocab_size, n_hidden) -> (batch_size, n_hidden)
dhpreact = (1.0 - h**2) * dh # h = torch.tanh(hpreact)
dbngain = (bnraw * dhpreact).sum(0, keepdim=True) # hpreact = bngain * bnraw + bnbias
dbnraw = bngain * dhpreact
dbnbias = dhpreact.sum(0)
'''
forward pass: bnraw = bndiff * bnvar_inv # (x_i - μ) / sqrt(σ^2 + ε)

dbndiff: direct element-wise operation, no need for summation

when multiply bnvar_inv (1,64) with bndiff (batch_size, 64), broadcasting allows this operation to apply bnvar_inv to each row of bndiff. The summation in the backward pass ensures that the gradient reflects the cumulative effect of bnvar_inv across the entire batch (boradcasting: applows operations between tensors of different shapes by )

dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
'''
dbndiff = bnvar_inv * dbnraw 
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
'''
Forward pass: bnvar_inv = (bnvar + 1e-5)**-0.5 # 1/sqrt(σ^2 + ε) 

Derivative of Inverse Square Root:
The term (bnvar + 1e-5)^-0.5 is used in the forward pass to compute the inverse square root of the variance (for normalization).
The derivative of x^-0.5 with respect to x is -0.5 * x^-1.5.

Numerical Stability:
The small constant 1e-5 is added to the variance to ensure numerical stability and prevent division by zero.

Gradient Calculation:
(-0.5 * (bnvar + 1e-5)^-1.5) is the derivative of the inverse square root with respect to bnvar.
'''
dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv

'''
forward pass: bnvar = 1/(batch_size-1) * (bndiff2).sum(0, keepdim=True) 

the sum operation reduce the tensor along a dimension
so its derivative is a tensor of ones with the same shape as the input to the sum
'''
dbndiff2 = (1.0/(batch_size-1)) * torch.ones_like(bndiff2) * dbnvar

'''
eariler we had: bnraw = bndiff * bnvar_inv 
bndiff2 = bndiff**2
Accumulation: since bndiff is involved in both operations, I need to accumulate the gradients from both paths -> dbndiff +=
'''
dbndiff += (2*bndiff) * dbndiff2
'''
forward pass: bndiff = hprebn - bnmeani 

1. the gradient of bndiff w.r.t bnmeani is -1
2. multiple dbndiff by -1 to get the gradient w.r.t bnmeani
3. since bnmeani is a mean across the batch, sum the gradients across the batch dimension (match the shape with bnmeani)
'''
dhprebn = dbndiff.clone()
dbnmeani = (-torch.ones_like(bndiff) * dbndiff).sum(0)
'''
forward pass: bnmeani = 1/batch_size * hprebn.sum(0, keepdim=True) 
'''
dhprebn += 1.0/batch_size * (torch.ones_like(hprebn) * dbnmeani)
'''
forward pass: hprebn = embcat @ W1 + b1
'''
dW1 = embcat.T @ dhprebn
dembcat = dhprebn @ W1.T
db1 = dhprebn.sum(0)
'''
forward pass: embcat = emb.view(emb.shape[0], -1)

view function is used to reshape a tensor without changing its data. The dembcat.view(emb.shape) operation is reshaping the dembcat tensor to have the same shape as the emb tensor.
'''
demb = dembcat.view(emb.shape)
'''
forward pass: emb = emb_matrix[Xbatch]

Xbatch.shape[0]: the number of samples
Xbatch.shape[1]: context length
Xbatch[row, col] retrieves the index of character for the current position

Accumulate graidents:
demb[row, col] contains the gradient of the loss w.r.t the embedding vector for the character at the current position
demb_matrix[char] += demb[row, col] updates the graident for the embedding vector corresponding to the character index by adding the gradient from the current sample and position. This accumulation is necessary because the same character can appear in multiple contexts across different samples in the batch
'''
demb_matrix = torch.zeros_like(emb_matrix)
for row in range(Xbatch.shape[0]):
   for col in range(Xbatch.shape[1]):
      char= Xbatch[row, col]
      demb_matrix[char] += demb[row, col]

cmp('logprobs', dlogprobs, logprobs)
cmp('probs', dprobs, probs)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
cmp('counts_sum', dcounts_sum, counts_sum)
cmp('counts', dcounts, counts)
cmp('norm_logits', dnorm_logits, norm_logits)
cmp('logit_maxs', dlogit_maxes, logit_maxes)
cmp('logits', dlogits, logits)
cmp('h',  dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)
cmp('hpreact', dhpreact, hpreact)
cmp('bngain', dbngain, bngain)
cmp('bnraw', dbnraw, bnraw)
cmp('bnbias', dbnbias, bnbias)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
cmp('bndiff2', dbndiff2, bndiff2)
cmp('bndiff', dbndiff, bndiff)
cmp('bnmeani', dbnmeani, bnmeani)
cmp('bhpreact', dhpreact, hpreact)
cmp('W1', dW1, W1)
cmp('b1', db1, b1)
cmp('embcat', dembcat, embcat)
cmp('emb', demb, emb)
cmp('emb_matrix', demb_matrix, emb_matrix)
# %%
hpreact.shape, bngain.shape, bnraw.shape, bnbias.shape
# %%
# Exercise 2: backprop through cross_entropy but all in one go
# to complete this challenge look at the mathematical expression of the loss,
# take the derivative, simplify the expression, and just write it out

# forward pass

# before:
# logit_maxes = logits.max(1, keepdim=True).values
# norm_logits = logits - logit_maxes # subtract max for numerical stability
# counts = norm_logits.exp()
# counts_sum = counts.sum(1, keepdims=True)
# counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
# probs = counts * counts_sum_inv
# logprobs = probs.log()
# loss = -logprobs[range(n), Yb].mean()

# now:
loss_fast = F.cross_entropy(logits, Ybatch)
print(loss_fast.item(), 'diff:', (loss_fast - loss).item())
# %%
# backward pass

# -----------------
# YOUR CODE HERE :)
dlogits = F.softmax(logits, 1)
dlogits[range(batch_size), Ybatch] -= 1
dlogits /= batch_size
# -----------------

cmp('logits', dlogits, logits) # I can only get approximate to be true, my maxdiff is 6e-9
# %%
'''
X-axis: represents the different classes (output neurons)
27 classes -> correspond to the characters in the vacab
Y-axis: represents the different samples in the batch
32 rows

White/Light: Higher positive gradient values.
Black/Dark: Higher negative gradient values.
Gray: Values closer to zero.

The gradients indicate how much each logit (output before softmax) should be adjusted to minimize the loss
- A higher positive gradient suggests increasing the logit for that class will reduce the loss
- A higher negative gradient suggests decreasing the logit for that class will reduce the loss
- Value close to zero indicate that small changes to the logit will not significantly affect the loss
'''
plt.figure(figsize=(8,8))
plt.imshow(dlogits.detach(), cmap='gray')
# %%
# Exercise 3: backprop through batchnorm but all in one go
# to complete this challenge look at the mathematical expression of the output of batchnorm,
# take the derivative w.r.t. its input, simplify the expression, and just write it out
# BatchNorm paper: https://arxiv.org/abs/1502.03167

# forward pass

# before:
# bnmeani = 1/n*hprebn.sum(0, keepdim=True)
# bndiff = hprebn - bnmeani
# bndiff2 = bndiff**2
# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
# bnvar_inv = (bnvar + 1e-5)**-0.5
# bnraw = bndiff * bnvar_inv
# hpreact = bngain * bnraw + bnbias

# now:
hpreact_fast = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias
print('max diff:', (hpreact_fast - hpreact).abs().max())
# %%
# backward pass

# before we had:
# dbnraw = bngain * dhpreact
# dbndiff = bnvar_inv * dbnraw
# dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
# dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
# dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
# dbndiff += (2*bndiff) * dbndiff2
# dhprebn = dbndiff.clone()
# dbnmeani = (-dbndiff).sum(0)
# dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)

# calculate dhprebn given dhpreact (i.e. backprop through the batchnorm)
# (you'll also need to use some of the variables from the forward pass up above)

# -----------------
# YOUR CODE HERE :)
'''
hpreact_fast = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias
'''
dhprebn = bngain*bnvar_inv/batch_size * (batch_size * dhpreact - dhpreact.sum(0) - batch_size/(batch_size-1)*bnraw*(dhpreact*bnraw).sum(0))
# -----------------

cmp('hprebn', dhprebn, hprebn) # I can only get approximate to be true, my maxdiff is 9e-10
# %%
# Exercise 4: putting it all together!
# Train the MLP neural net with your own backward pass

# init
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
logging.info(f'Number of parameters: {sum(p.nelement() for p in parameters)}')
for p in parameters:
  p.requires_grad = True

# same optimization as last time
max_steps = 20000
batch_size = 32
n = batch_size # convenience
lossi = []

# use this context manager for efficiency once your backward pass is written (TODO)
#with torch.no_grad():

# kick off optimization
with torch.no_grad():
    for i in range(max_steps):
        # if i % 10000 == 0:
        #     logging.info(f"Step {i}/{max_steps}")

        # minibatch construct
        ix = torch.randint(0, Xtrain.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtrain[ix], Ytrain[ix]  # batch X,Y

        # forward pass
        emb = C[Xb]  # embed the characters into vectors
        embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors
        # if i % 10000 == 0:
        #   logging.debug(f"Step {i}: Forward pass - Embedding shape: {emb.shape}, Concatenated shape: {embcat.shape}")

        # Linear layer
        hprebn = embcat @ W1 + b1  # hidden layer pre-activation
        # if i % 10000 == 0:
        #   logging.debug(f"Step {i}: Forward pass - hprebn shape: {hprebn.shape}")

        # BatchNorm layer
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar = hprebn.var(0, keepdim=True, unbiased=True)
        bnvar_inv = (bnvar + 1e-5) ** -0.5
        bnraw = (hprebn - bnmean) * bnvar_inv
        hpreact = bngain * bnraw + bnbias
        # if i % 10000 == 0:
        #   logging.debug(f"Step {i}: Forward pass - BatchNorm mean: {bnmean.shape}, variance: {bnvar.shape}")

        # Non-linearity
        h = torch.tanh(hpreact)  # hidden layer
        logits = h @ W2 + b2  # output layer
        loss = F.cross_entropy(logits, Yb)  # loss function
        # if i % 10000 == 0:
        #   logging.info(f"Step {i}: Loss = {loss.item()}")

        # backward pass
        for p in parameters:
            p.grad = None

        # Step 1: Backprop through output layer
        dlogits = F.softmax(logits, 1)
        dlogits[range(n), Yb] -= 1
        dlogits /= n
        # if i % 10000 == 0:
        #   logging.debug(f"Step {i}: Backward pass - Step 1: dlogits shape: {dlogits.shape}")

        # Step 2: Backprop through 2nd layer
        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)
        # if i % 10000 == 0:
        #   logging.debug(f"Step {i}: Backward pass - Step 2: dW2 shape: {dW2.shape}, db2 shape: {db2.shape}")

        # Step 3: Backprop through tanh
        dhpreact = (1.0 - h ** 2) * dh  # h = tanh(hpreact)
        # if i % 10000 == 0:
        #   logging.debug(f"Step {i}: Backward pass - Step 3: dhpreact shape: {dhpreact.shape}")

        # Step 4: Backprop through batchnorm
        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        dhprebn = bngain * bnvar_inv / n * (n * dhpreact - dhpreact.sum(0) - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0))
        # if i % 10000 == 0:
        #   logging.debug(f"Step {i}: Backward pass - Step 4: dbngain shape: {dbngain.shape}, dbnbias shape: {dbnbias.shape}, dhprebn shape: {dhprebn.shape}")

        # Step 5: Backprop through 1st layer
        dembcat = dhprebn @ W1.T
        dW1 = embcat.T @ dhprebn
        db1 = dhprebn.sum(0)
        # if i % 10000 == 0:
        #   logging.debug(f"Step {i}: Backward pass - Step 5: dW1 shape: {dW1.shape}, db1 shape: {db1.shape}")

        # Step 6: Backprop through embedding
        demb = dembcat.view(emb.shape)
        dC = torch.zeros_like(C)
        for row in range(Xb.shape[0]):
            for col in range(Xb.shape[1]):
                char_index = Xb[row, col]
                dC[char_index] += demb[row, col]
        # if i % 10000 == 0:
        #   logging.debug(f"Step {i}: Backward pass - Step 6: dC shape: {dC.shape}")

        grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

        # update
        lr = 0.1 if i < 100000 else 0.01  # step learning rate decay
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad

        # track stats
        if i % 10000 == 0:
            logging.info(f'Step {i:7d}/{max_steps:7d}: Loss = {loss.item():.4f}')
        lossi.append(loss.log10().item())

        # if i >= 100: # Early break for testing
        #    break
# %%
# calibrate the batch norm at the end of training

with torch.no_grad():
  # pass the training set through
  emb = C[Xtrain]
  embcat = emb.view(emb.shape[0], -1)
  hpreact = embcat @ W1 + b1
  # measure the mean/std over the entire training set
  bnmean = hpreact.mean(0, keepdim=True)
  bnvar = hpreact.var(0, keepdim=True, unbiased=True)

# %%
# evaluate train and val loss

@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (Xtrain, Ytrain),
    'val': (Xdev, Ydev),
    'test': (Xtest, Ytest),
  }[split]
  emb = C[x] # (N, block_size, n_embd)
  embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  hpreact = embcat @ W1 + b1
  hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
  h = torch.tanh(hpreact) # (N, n_hidden)
  logits = h @ W2 + b2 # (N, vocab_size)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')

# %%
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      # forward pass
      emb = C[torch.tensor([context])] # (1,block_size,d)      
      embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
      hpreact = embcat @ W1 + b1
      hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
      h = torch.tanh(hpreact) # (N, n_hidden)
      logits = h @ W2 + b2 # (N, vocab_size)
      # sample
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))
# %%
