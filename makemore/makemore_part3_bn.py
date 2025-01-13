# %% [markdown]
## makemore: part 3
'''
Custom functions: mean, std (refer to line:267)
'''
def custom_mean(tensor, dim, keepdim=False):
    # Calculate the mean along the specified dimension
    if dim == 0:
        # Mean across rows (for each column)
        mean_values = [sum(col) / len(col) for col in zip(*tensor)]
    elif dim == 1:
        # Mean across columns (for each row)
        mean_values = [sum(row) / len(row) for row in tensor]
    else:
        raise ValueError("Invalid dimension specified. Use 0 or 1.")

    if keepdim:
        # Retain the reduced dimension
        if dim == 0:
            return [[val] for val in mean_values]
        elif dim == 1:
            return [[val] * len(tensor[0]) for val in mean_values]
    else:
        return mean_values

def custom_std(tensor, dim, keepdim=False):
    # Calculate the mean first
    mean_values = custom_mean(tensor, dim, keepdim=False)

    # Calculate the standard deviation along the specified dimension
    if dim == 0:
        # Std across rows (for each column)
        std_values = [
            (sum((x - mean) ** 2 for x in col) / len(col)) ** 0.5
            for col, mean in zip(zip(*tensor), mean_values)
        ]
    elif dim == 1:
        # Std across columns (for each row)
        std_values = [
            (sum((x - mean) ** 2 for x in row) / len(row)) ** 0.5
            for row, mean in zip(tensor, mean_values)
        ]
    else:
        raise ValueError("Invalid dimension specified. Use 0 or 1.")

    if keepdim:
        # Retain the reduced dimension
        if dim == 0:
            return [[val] for val in std_values]
        elif dim == 1:
            return [[val] * len(tensor[0]) for val in std_values]
    else:
        return std_values

# %%
import torch
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt 
# %%
# read in all the words
words = open('names.txt', 'r').read().splitlines()
print(f"first 8 words: {words[:8]}")
# %%
# data description 
print(f"number of words: {len(words)}")
# %%
# build the vocabulary of characters and mapping to/from integers
chars = sorted(list(set(''.join(words))))
print(f"sorted all the chars in words: {chars}")
print(f"number of all chars in words: {len(chars)}")

char_to_index = {ch:i+1 for i, ch in enumerate(chars)}
print(f"char to integer map: {char_to_index}")
# add . to chat to integer map
char_to_index["."] = 0

index_to_char = {i:ch for ch, i in char_to_index.items()}
vocab_size = len(index_to_char)
print("int to char map: ")
print(index_to_char)
print(f"size of vocabulary: {vocab_size}")
# %%
# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w:
            idx = char_to_index[ch]  
            print(f"Context: Input sequence {context} {[index_to_char[c] for c in context]}")
            print(f"The index of next char for the input sequence '{ch}': {idx} (output)")
            X.append(context)
            Y.append(idx)
            context = context[1:] + [idx]
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Y.shape}")
    return X, Y
# %%
random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtrain, Ytrain = build_dataset(words[:n1]) # 80%
Xdev, Ydev = build_dataset(words[n1:n2]) # 10%
Xtest, Ytest = build_dataset(words[n2:]) # 10%
# %%
# MLP revisited
n_embeddings = 10 # the dimenionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducbility
embedding_matrix = torch.randn((vocab_size, n_embeddings), generator=g) # (27, 10)
W1 = torch.randn((n_embeddings * block_size, n_hidden), generator=g) * 0.01 # (30, 200)
b1 = torch.randn(n_hidden, generator=g) * 0.1 # (200)
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01 # (200, 27)
b2 = torch.randn(vocab_size, generator=g) * 0.1 # (27)

# %%
parameters = [embedding_matrix, W1, W2, b2]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
    p.requires_grad = True

# %%
embedding_mapping = embedding_matrix[Xtrain] # or indexing
embedding_mapping.shape
# %%
max_steps = 200000
batch_size = 32
track_loss = []

for i in range(max_steps):
    # minibatch construct
    indices = torch.randint(0, Xtrain.shape[0], (batch_size, ), generator=g)
    Xbatch, Ybatch = Xtrain[indices], Ytrain[indices] # batch X, Y

    # forward pass
    # (N, block_size, n_embeddings)
    embedding_mapping = embedding_matrix[Xbatch] # or indexing
    # concat into (N, block_size * n_embeddings) (32, 3 * 10) - flatten
    embedding_concatenate = embedding_mapping.view(embedding_mapping.shape[0], embedding_mapping.shape[1] * embedding_mapping.shape[2]) 
    # hidden layer (32, 20) - embeddings (32, 30) W1 (30, 200) b1 (200)
    hidden_layer_pre_activation = embedding_concatenate @ W1 + b1
    hidden_layer = torch.tanh(hidden_layer_pre_activation)
    # output layer (32, 27) - hidden layer (32, 200) W2 (200, 27) b2 (27)
    logits = hidden_layer @ W2 + b2 
    # softmax
    '''
    cross_entropy: combines log_softmax and negative log likelihood loss in on single function
    used to compute the loss between the predicted class probabilities and the true class labels
    '''
    loss = F.cross_entropy(logits, Ybatch)

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    learning_rate = 0.1 if i < 100000 else 0.01 # step learaning rate decay
    for p in parameters:
        p.data += -learning_rate * p.grad
    
    # track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item(): 4f}")
    
    track_loss.append(loss.log10().item())
    break
# %%
'''
concat 2D tensor (32, 200) into 1D tensor (6400)
for example
[[1,2,3,...,200],[201,...,400],...(30 more rows)] -> [1,2,3,...,200,201,...,401,...,6400]
'''
# hidden_layer.view(hidden_layer.shape[0] * hidden_layer.shape[1]).tolist()
hidden_layer.view(-1).tolist()
# %%
plt.hist(hidden_layer_pre_activation.view(-1).tolist(), 50)
# %%
plt.hist(hidden_layer.view(-1).tolist(), 50)
# %%
hidden_layer_pre_activation
# %%
hidden_layer
# %%
'''
Each pixel represents whether a particular neuron in the hidden layer is highly activated (absolute value greater than 0.99).
White pixels (True) indicate neurons with activations exceeding the threshold.
Black pixels (False) indicate neurons with lower activations.

Highly activated neuron indicates that the neuron is responding strongly to a certain input. However, if many neurons are consistently saturated (close to -1 or 1), the gradients during backpropagation become very small, which can:
Slow learning: The gradients near saturation are close to 0, causing negligible weight updates.
Cause vanishing gradients: This makes it harder for the network to learn effectively, especially in deeper layers.

the reaons for highly activated neruon: 
before tanh (activate), hidden_layer_pre_activation has extreme large number
plt.hist(hidden_layer_pre_activation.view(-1).tolist(), 50)
after tanh(x), those large number would get very close to 1 or -1 (also mean saturated) (check tanh graph) -> highly activated

https://chatgpt.com/c/677acf8a-3ac4-8004-8af5-166ea32676ff (include tanh graph)

after regularize (add penalty) to paramters, greatly reduce saturated neuron
W1 = torch.randn((n_embeddings * block_size, n_hidden), generator=g) * 0.01 # (30, 200)
b1 = torch.randn(n_hidden, generator=g) * 0.1 # (200)
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01 # (200, 27)
b2 = torch.randn(vocab_size, generator=g) * 0.1 # (27)

another solution: Kai-Ming Normal Distribution 
https://chatgpt.com/c/677ad8a8-6928-8004-8199-2ff58cefd301
but it is not important after batch normalization came out

another good solution: Batch Normalization
'''
plt.figure(figsize=(20,10))
plt.imshow(hidden_layer.abs() > 0.99, cmap='gray', interpolation='nearest')
# %%
plt.plot(track_loss)

# %% [markdown]
## Batch Normalization
'''
Batch Normalization Purpose:
Normalize the input to each layer to improve training stability and speed up convergence 
It helps mitigate issues like internal covariate shift

What is Covergence?
it's when the algorithm's output stabilizes and stops changing significantly with further iterations.

What is Internal Covariate?
Internal covariate shift refers to the phenomenon where the distribution of inputs to a layer in a neural network changes during training. 
This shift occurs because the parameters of the previous layers are updated continuously, which in turn changes the input distribution for subsequent layers. 
This can slow down the training process and make it harder for the network to converge.
'''
# %%
# MLP revisited
n_embeddings = 10 # the dimenionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducbility
embedding_matrix = torch.randn((vocab_size, n_embeddings), generator=g) # (27, 10)
W1 = torch.randn((n_embeddings * block_size, n_hidden), generator=g) * 0.01 # (30, 200)
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01 # (200, 27)
b2 = torch.randn(vocab_size, generator=g) * 0.1 # (27)

# batch normalization parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
'''
During training, the mean and standard deviation are computed for each mini-batch. 
However, during inference (testing), you want to use a stable estimate of these statistics that is representative of the entire training dataset.
bnmean_running and bnstd_running are used to accumulate a running average of the batch-wise mean and standard deviation. 
This running average is then used during inference to normalize the activations.
3. Why Running Estimates?
Using the running estimates during inference ensures that the model's behavior is consistent and not dependent on the specific batch of data it sees during testing.
'''
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [embedding_matrix, W1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
    p.requires_grad = True
# %%
max_steps = 200000
batch_size = 32
track_loss = []

for i in range(max_steps):

    # minibatch
    indices = torch.randint(0, Xtrain.shape[0], (batch_size, ), generator=g)
    Xbatch, Ybatch = Xtrain[indices], Ytrain[indices]

    # forward pass
    # embed the characters into vectors - indexing operation
    '''
    embedding_matrix is a tensor where each row corresponds to an embedding vector for a character in your vocabulary.
    Xbatch is a tensor containing indices that correspond to characters in your vocabulary.
    By using embedding_matrix[Xbatch], you are effectively using the indices in Xbatch to select the corresponding rows (embedding vectors) from embedding_matrix.
    '''
    embedding_mapping = embedding_matrix[Xbatch] 
    # concatenate the vectors from (batch_size, block_size, n_embeddings) to (batch_size, block_size * n_embeddings)
    # embedding_mapping.view(embedding_mapping.shape[0], embedding_mapping.shape[1] * embedding_mapping.shape[2])
    embedding_concat = embedding_mapping.view(embedding_mapping.shape[0], -1) 

    # Linear layer
    hidden_layer_pre_activation = embedding_concat @ W1 # + b1

    # Batch Normalization layer
    # --------------------------------
    # normalize: output=(input-mean)/std
    '''
    why use mean and standard deviation -> normalize the activation 
    normalization helps in maintaining a stable distribution of activations -> lead to faster convergence and improved performance
    '''
    bnmeani = hidden_layer_pre_activation.mean(0, keepdim=True)
    bnstdi = hidden_layer_pre_activation.std(0, keepdim=True)
    normalized_hidden_layer_pre_activation = (hidden_layer_pre_activation - bnmeani) / bnstdi
    # scale and shift: output=γ×normalized_input+β
    hidden_layer_pre_activation_bn = bngain * normalized_hidden_layer_pre_activation + bnbias

    '''
    why use torch.no_grad()?
    not require gradient computation 
    bnmean_running and bnstd_running are not parameters and will be used in the later evaluation because they are reflecting the overall data distribution learned during training
    '''
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
    # --------------------------------

    # Non-linear activation
    hidden_layer = torch.tanh(hidden_layer_pre_activation_bn) # hidden layer
    logits = hidden_layer @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Ybatch) # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    learning_rate = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -learning_rate * p.grad
    
    # track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    track_loss.append(loss.log10().item())

# %%
'''
bnmean and bnstd are not used anywhere 
instead bnmean_running and bnstd_running are using in the later validation and test
'''
with torch.no_grad():
    # pass the training set through
    embedding_mapping = embedding_matrix[Xtrain]
    embedding_concat = embedding_mapping.view(embedding_mapping.shape[0], -1)
    hidden_layer_preact = embedding_concat @ W1 + b1
    # measure the mean/std over the entire training set
    bnmean = hidden_layer_preact.mean(0, keepdim=True)
    bnstd = hidden_layer_preact.std(0, keepdim=True)
# %%
@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
    x, y = {
        'train': [Xtrain, Ytrain],
        'validation': [Xdev, Ydev],
        'test': [Xtest, Ytest]
    }[split]

    # embedding 
    embedding_mapping = embedding_matrix[x] 
    embedding_concatenate = embedding_mapping.view(embedding_mapping.shape[0], -1) # concat into (N, block_size * n_embeddings)

    # linear layer
    hidden_layer_pre_activation = embedding_concatenate @ W1 # + b1

    # batch normalization
    # -------------------
    normalized_hidden_layer_pre_activation = (hidden_layer_pre_activation - bnmean_running) / bnstd_running
    # scale and shift: output=γ×normalized_input+β
    hidden_layer_pre_activation_bn = bngain * normalized_hidden_layer_pre_activation + bnbias
    # -------------------

    hidden_layer = torch.tanh(hidden_layer_pre_activation_bn)
    logits = hidden_layer @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('validation')
# %%
# sample from the model
g = torch.Generator().manual_seed(2147483647) # for reproducbility

for _ in range(1):
    out = []
    context = [0] * block_size
    while True:
        # forward pass

        # (1, block_size, n_embeddings)
        x = torch.tensor([context])

        # embedding 
        # ----------------------
        # mapping (or indexing) 
        embedding_mapping = embedding_matrix[x]
        # concatenation (or flatten): (1, block_size, n_embeddings) -> (1, block_size * n_embeddings)
        embedding_concatenate = embedding_mapping.view(embedding_mapping.shape[0], -1)
        # ----------------------

        # linear layer: after use batch normalization, stop using (+ bias1) because we add bnbiad later
        hidden_layer_pre_activation = embedding_concatenate @ W1 # + b1

        # batch normalization 
        # ----------------------
        normalized_hidden_layer_pre_activation = (hidden_layer_pre_activation - bnmean_running) / bnstd_running
        # scale and shift
        hidden_layer_pre_activation_bn = bngain * normalized_hidden_layer_pre_activation + bnbias
        # ----------------------

        # non-linear
        hidden_layer = torch.tanh(hidden_layer_pre_activation_bn)
        logits = hidden_layer @ W2 + b2  # output layer
        '''
        softmax: convert raw logits (unnormalized scores) into probabilities
        ensure the output is probability distribution (all values are between 0 & 1 and sum to 1)
        intepreting the logits as probabilities of each character being the next in the sequence.

        Softmax calculation: 
        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdim=True)
        loss = -prob[torch.arange(32), Y].log().mean()
        '''
        probs = F.softmax(logits, dim=1)

        '''
        sample an index from probability distribution
        '''
        idx = torch.multinomial(probs, num_samples=1, generator=g).item()

        # shift the context window and track the samples
        context = context[1:] + [idx]

        out.append(idx)
        if idx == 0:
            break
    print(''.join(index_to_char[i] for i in out))
# %% [markdown]
## Let's train a deeper network
### reproduce nn.Module 

'''
fan_in: number of input
fan_out: number of output
'''
class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        '''
        why divide by fan_in ** 0.5 (the square root of the number input)?
        
        Variance scaling:
        scale the weights -> ensures the variance of the outputs of each layer is approximately the same as the variance of its inputs

        Avoiding vanishing/exploding gradients:
        prevent the gradients from becoming too small (vanishing graidents) 
        or too large (exploding gradients) 

        Balanced initialization: keep the activations and gradients in a reasonable range -> faster convergence and stable training

        BTW, scaling refers to adjusting the inital values of the weights to ensure that network's activations and gradients remain within a resonable range during the inital stages of training
        '''
        self.weight =  torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out =  x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps # eps (ϵ) epsilon
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim) # bngain
        self.beta = torch.zeros(dim) # bnbias
        # buffers (trained with a running 'momentum update')
        # mean = 0 and variance = 1 distribution
        '''
        momentum is used to control the update of the running estimates of the mean and variance
        running estimates are used during testing to normalize the input data
        '''
        self.momentum = momentum
        self.running_mean = torch.zeros(dim) 
        self.running_var = torch.ones(dim) 

    def __call__(self, x):
        # calcualte the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = self.var(0, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        '''
        add eps (ϵ) to the variable -> prevent division by 0
        when the variance is very small or zero, dividing by standard deviation (square root the variance) can lead to extremelyt large values
        adding a small constant eps (ϵ) ensures that the denominator is never zero
        '''
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = self.running_mean * (1 - self.momentum) + xmean * self.momentum
                self.running_var = self.running_var * (1 - self.momentum) + xvar * self.momentum
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
# %%
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 100 # the number of neruons in the hidden layer of the MLP
g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((vocab_size, n_embd), generator=g)
layers = [
    Linear(n_embd * block_size, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
]

with torch.no_grad():
    #  last layer: make less confident
    layers[-1].weight *= 0.1
    # all other layers: apply gain
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            '''
            why 5/3 -> Xavier initialization
            5/3 factor is often associated with the initialization strategy for networks using the tanh activation function
            it helps maintain the variance of activations across layers -> increase convergence

            keep the activation and gradients in a reasonable range 
            preventing them become too small (vanishing graidents) or too large (exploding graidents)
            '''
            layer.weight *= 5/3 # 1
            '''
            interesting phenomenon: layer.weight *= 1 (not adjusting the initial weight)
            gradient distribtion becomes less centered and more spread out
            several implications:
            1. variance of activations: a more evenly distributed gradient might indicate that the activaqtions are not as well-scaled across layers
            2. training stability: gradient too large (explding gradient) or too small (vanishing graident)
            3. convergence speed: slow down convergence -> the model take longer to find the optimal paramter values
            4. model performance: model might struggle to learn complex patterns if gradients are not well-scaled
            '''

'''
some addon knowledge:
when we say activations and gradients are 'well-scaled', 
it means they are within a reasonable range -> effective learning

1. Activation
- should not be too large or too small
- for activation functions like tanh or sigmoid, inputs should ideally be around zero to avoid saturation

2. Gradients
- should be large engough to make meaningful updates to the parameters
- should not be so large -> instability or divergence
'''
parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True
# %%
max_steps = 200000
batch_size = 32
track_loss = []
ud = []

for i in range(max_steps):

    # minibatch
    '''
    not going to train the entire train dataset just select randomly a minibatch of data for each iteration
    '''
    indices = torch.randint(0, Xtrain.shape[0], (batch_size, ), generator=g) # array of index: length = batch_size
    Xbatch, Ybatch = Xtrain[indices], Ytrain[indices] # batch X, Y

    # forward pass
    emb = C[Xbatch] # embed the characters into vectors -> indexing or mapping
    x = emb.view(emb.shape[0], -1) # concatenate the vectors
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Ybatch) # loss function

    # backward pass
    for layer in layers:
        layer.out.retain_grad() 
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad
    
    # track stats
    if i % 10000 == 0: # print every one in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    track_loss.append(loss.log10().item())

    with torch.no_grad():
        ud.append([(lr*p.grad.std() / p.data.std()).log10().item() for p in parameters])

    if i >= 1000:
        break  
# %%
# visualize histograms
plt.figure(figsize=(20,4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
    if isinstance(layer, Tanh):
        t = layer.out
        print(f'layer {i} {layer.__class__.__name__:10s}: mean {t.mean():+.2f}, std {t.std():.2f}')
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'later {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('gradient distribution')
# %%
'''
remind a bit:
graidients are used to adjust the model parameters (like weight and bias) to minizie the difference between the predicted values and the actual labels
'''
plt.figure(figsize=(20,4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out.grad
        print(f"layer {i} ({layer.__class__.__name__:10s}): mean {t.mean():+f}, std {t.std():e}")
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer.__class__.__name__})")
plt.legend(legends)
plt.title("gradient distribution")
# %%
plt.figure(figsize=(20,4)) # width and height of the plot
legends = []
for i, p in enumerate(parameters):
    t = p.grad
    if p.ndim == 2:
        print(f"weight {tuple(p.shape)} | mean {t.mean():+f} | std {t.std():e} | grad:data ratio {(t.std() / p.std()):e}")
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"{i} {tuple(p.shape)}")
plt.legend(legends)
plt.title("weights gradient distribution")
# %%
plt.figure(figsize=(20,4))
legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2:
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append(f'param {i}')
plt.plot([0, len(ud)], [-3, -3], 'k') 
plt.legend(legends)
# %%
