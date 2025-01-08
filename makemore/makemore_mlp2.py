# %% [markdown]
## makemore: part 3
# %%
import torch
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
            print(f"The index of char '{ch}': {idx} (output)")
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
for index, char_embedding in enumerate(embedding_matrix):
    char = index_to_char[index]
    print(f"char '{char}' correspond to embedding vector: {char_embedding}")
W1 = torch.randn((n_embeddings * block_size, n_hidden), generator=g) * 0.01 # (30, 200)
b1 = torch.randn(n_hidden, generator=g) * 0.1 # (200)
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01 # (200, 27)
b2 = torch.randn(vocab_size, generator=g) * 0.1 # (27)

# %%
parameters = [embedding_matrix, W1, b1, W2, b2]
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
# %%
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

    embedding_mapping = embedding_matrix[x] 
    embedding_concatenate = embedding_mapping.view(embedding_mapping.shape[0], -1) # concat into (N, block_size * n_embeddings)
    hidden_layer = torch.tanh(embedding_concatenate @ W1 + b1)
    logits = hidden_layer @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('validation')
# %%
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10) # for reproducbility

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        # forward pass the nerual net
        # (1, block_size, n_embeddings)
        x = torch.tensor([context])
        embedding_mapping = embedding_matrix[x]
        # (1, block_size, n_embeddings) -> (1, block_size * n_embeddings)
        embedding_concatenate = embedding_mapping.view(embedding_mapping.shape[0], embedding_mapping.shape[1] * embedding_mapping.shape[2])
        hidden_layer = torch.tanh(embedding_concatenate @ W1 + b1)
        logits = hidden_layer @ W2 + b2 
        '''
        softmax: convert raw logits (unnormalized scores) into probabilities
        exponentitates each logit and then normalizes them by dividing by the sume of all exponentiated logits
        ensure the output is probability distribution (all values are between 0 & 1 and sum to 1)

        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdim=True)
        loss = -prob[torch.arange(32), Y].log().mean()
        '''
        probs = F.softmax(logits, dim=1)
        # sample from distribution
        idx = torch.multinomial(probs, num_samples=1, generator=g).item()
        # shift the context window and track the samples
        context = context[1:] + [idx]
        out.append(idx)
        if idx == 0:
            break
    print(''.join(index_to_char[i] for i in out))
# # %%
# x = torch.randn(1000, 10)
# # 10 ** 0.5 == sqrt(10)
# w = torch.randn(10, 200) / 10**0.5 
# y = x @ w
# print(x.mean(), x.std())
# print(w.mean(), w.std())
# # %%
# plt.figure(figsize=(20,5))
# plt.subplot(121)
# plt.hist(x.view(-1).tolist(), 50, density=True)
# plt.subplot(122)
# plt.hist(y.view(-1).tolist(), 50, density=True)

# %%
# batch normalization gain & loss
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [embedding_matrix, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
    p.requires_grad = True
# %%
# Batch Normalization (refer to discussion line:172 - dealing with gradient vanish and neuron saturation (highly activated))
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
    hidden_layer_pre_activation = embedding_concat @ W1 + b1

    bnmeani = hidden_layer_pre_activation.mean(0, keepdim=True)
    bnstdi = hidden_layer_pre_activation.std(0, keepdim=True)

    hidden_layer_pre_activation = bngain * (hidden_layer_pre_activation - bnmeani) / bnstdi + bnbias

    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    hidden_layer = torch.tanh(hidden_layer_pre_activation) # hidden layer

    logits = hidden_layer @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Ybatch)

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
