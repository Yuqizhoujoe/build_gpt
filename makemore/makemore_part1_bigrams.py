# %% [markdown]
## Makemore
'''
a tensor is a multi-dimensional array: scalar, vector, matrix

Dimensions:
Scalar: A single numbger (0-dimensional)
Vector: A list of numbers (1-dimensional)
Matrix: A 2-dimensional array (2-dimensional)
Tensor: A n-dimensional array (n-dimensional)

Data Types:
Tensors can hold data of various types, including integers, floating-point numbers, and complex numbers.
'''
# %%
words = open("names.txt", "r").read().splitlines()
words[:10]
len(words)
# %%
print(f"Min word length: {min(len(word) for word in words)}")
print(f"Max word length: {max(len(word) for word in words)}")
# %%
'''
A bigram is a sequence of two adjacent elements from a string of tokens, which can be letters, syllables, or words. In the context of text processing and natural language processing (NLP), bigrams are often used to analyze the frequency of pairs of words or characters appearing together in a text.
For example, in the word "hello", the character bigrams are:
<S>h (assuming <S> is a start token)
he
el
ll
lo
o<E> (assuming <E> is an end token)
Bigrams are useful for understanding the structure and patterns within text data, such as predicting the next character or word in a sequence, or for building language models. In your code, bigrams are used to create a frequency matrix that helps in generating new sequences based on the learned patterns.
'''
bigram_dict = {}
for word in words:
      chs = ['<S>'] + list(word) + ['<E>']
      '''
      chs: ['<S>', 'e', 'm', 'm', 'a', '<E>']
      chs[1:]: ['e', 'm', 'm', 'a', '<E>']
      '''
      for ch1, ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            bigram_dict[bigram] = bigram_dict.get(bigram, 0) + 1
bigram_dict.items()
# %%
sorted(bigram_dict.items(), key=lambda x: x[1], reverse=True)

# %%
import torch
# %%
charSet = set(''.join(words))
charList = list(charSet)
sortedCharList = sorted(charList)
sortedCharList
# %%
# charToIndex = {}
# for i, char in enumerate(sortedCharList):
#       charToIndex[char] = i
# charToIndex
charToIndex = {ch: i+1 for i, ch in enumerate(sortedCharList)}
# charToIndex['<S>'] = 26
# charToIndex['<E>'] = 27
charToIndex['.']  = 0
charToIndex
# %%
'''
why 27x27?
27 letters + .
'''
N = torch.zeros((27,27), dtype=torch.int32)
for word in words:
      # chars = ['<S>'] + list(word) + ['<E>']
      chars = ['.'] + list(word) + ['.']
      for ch1, ch2 in zip(chars, chars[1:]):
            idx1 = charToIndex[ch1]
            idx2 = charToIndex[ch2]
            N[idx1, idx2] += 1
# %%
indexToChar = {i: ch for ch, i in charToIndex.items()}
indexToChar
# %%
N[3,3].item()
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
      for j in range(27):
            charStr = indexToChar[i] + indexToChar[j]
            plt.text(j, i, charStr, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i,j].item(), ha="center", va="top", color="gray")
plt.axis('off');

# %%
'''
# The `torch.multinomial` function samples indices from the input probability distribution `probabilities`.
# It returns a tensor of 100 indices, allowing for replacement, meaning the same index can be chosen multiple times.
# The returned indices indicate which elements from `probabilities` were sampled, based on their probabilities.

# Example: If `probabilities` is [0.6064, 0.3033, 0.0903], index 0 is more likely to appear in the result than indices 1 or 2.

# Set a random generator for reproducibility
g = torch.Generator().manual_seed(2147483647)

# Define a probability distribution
rand_numbers = torch.rand(3, generator=g) 
probabilities = rand_numbers / rand_numbers.sum() # [0.6064, 0.3033, 0.0903]

# Sample 100 indices from the distribution
sampled_indices = torch.multinomial(probabilities, num_samples=100, replacement=True, generator=g)

# Print the sampled indices
print("Sampled indices:", sampled_indices.tolist())

# In this example, index 0 should appear more frequently in the output.

'''
import torch
# seed is like the key
# the same seed will always generate the same random numbers -> reproducibility
# different seeds will generate different random numbers -> randomness
# {seed: random numbers}
g = torch.Generator().manual_seed(2147483647)
rand_numbers = torch.rand(3, generator=g)
probabilities = rand_numbers / rand_numbers.sum()
torch.multinomial(probabilities, num_samples=100, replacement=True, generator=g)

# %% 
'''
the frequency pf bigrams (pair of characters)
'''
bigram_matrix = (N+1).float()
print(bigram_matrix.shape)
# print(bigram_matrix)

'''
sum across the columns for each row
'''
bigram_matrix_sum = bigram_matrix.sum(1, keepdim=True)
print(bigram_matrix_sum.shape)
# print(bigram_matrix_sum)

'''
normalize the frequency of bigrams
'''
bigram_matrix_normalized = bigram_matrix / bigram_matrix_sum
print(bigram_matrix_normalized.shape)
print(bigram_matrix_normalized[0])

'''
We sum across the columns for each row to get the total frequency count for each row. This helps us understand the total occurrences of bigrams starting with each character.

For example, if we have a bigram frequency matrix like this:
[[2, 3, 5],
 [1, 4, 2],
 [3, 2, 1]]

Summing across the columns for each row gives us:
[[10],
 [7],
 [6]]

Next, we normalize the frequency of bigrams by dividing each element in a row by the sum of that row. This converts the frequency counts into probabilities, which makes it easier to work with and compare.

Using the previous example, the normalized matrix would be:
[[0.2, 0.3, 0.5],
 [0.1429, 0.5714, 0.2857],
 [0.5, 0.3333, 0.1667]]

This normalization ensures that the sum of probabilities in each row is 1, making it a valid probability distribution.
'''

# %% 
'''
The Normalization Broadcasting process
'''
import random
def generate_2d_array(rows, cols):
      return [[random.randint(1, 10) for _ in range(cols)] for _ in range(rows)]
matrix = generate_2d_array(10, 10)

def sum_2d_array_by_cols(matrix):
      col_sums = []
      for row in matrix:
            sum = 0
            for col in row:
                  sum += col
            col_sums.append(sum)
      return col_sums
col_sums = sum_2d_array_by_cols(matrix)

def normalize_2d_array_by_cols(matrix):
      print("matrix:")
      for row in matrix:
            print(row)
      col_sums = sum_2d_array_by_cols(matrix)
      print(f"col_sums: {col_sums}")
      normalized_matrix = []
      for row in matrix:
            normalized_row = []
            for col, col_sum in zip(row, col_sums):
                  normalized_row.append(round(col / col_sum, 4))
            normalized_matrix.append(normalized_row)
      return normalized_matrix

normalized_matrix = normalize_2d_array_by_cols(matrix)
print("normalized_matrix:")
for row in normalized_matrix:
      print(row)

# %%
g = torch.Generator().manual_seed(2147483647)

for i in range(20):
      out = []
      index = 0
      while True:
            prob = bigram_matrix_normalized[index]
            index = torch.multinomial(prob, num_samples=1, replacement=True, generator=g).item()
            out.append(indexToChar[index])
            if index == 0:
                  break
      print(''.join(out))
      
      
# %%
'''
Why use log probability?

a*b*c
log(a*b*c) = log(a) + log(b) + log(c)

so we can add up the log probabilities instead of multiplying the probabilities
'''
# %%      
def get_normalized_neg_log_likelihood(words):
      log_likelihood = 0.0
      n = 0
      for word in words:
            chars = ['.'] + list(word) + ['.']
            for char1, char2 in zip(chars, chars[1:]):
                  index1 = charToIndex[char1]
                  index2 = charToIndex[char2]
                  prob = bigram_matrix_normalized[index1, index2]
                  log_prob = torch.log(prob)
                  log_likelihood += log_prob
                  n += 1
                  print(f"{char1}{char2}: {prob:.4f} {log_prob:.4f}")
                  
      return -log_likelihood/n

nll = get_normalized_neg_log_likelihood(words[:2])
print(f"nll: {nll:.4f}")
# %% [markdown]
'''
Maximum Likelihood Estimation

maximize likelihood of the data w.r.t. the model parameters
equivalent to maximizing the log likelihood (because log is monotonic)
equivalent to minimizing the negative log likelihood
equivalent to minimizing the average negative log likelihood
'''
# %%
# Creating the training set of bigrams (x,y)

xs, ys = [], []

for word in words[:1]:
      chars = ['.'] + list(word) + ['.']
      for char1, char2 in zip(chars, chars[1:]):
            index1 = charToIndex[char1]
            index2 = charToIndex[char2]
            print(f"input: {char1}, output: {char2}")
            xs.append(index1)
            ys.append(index2)
            
xs_tensor = torch.tensor(xs)
ys_tensor = torch.tensor(ys)

# %%
xs_tensor
# %%
ys_tensor
# %%
'''
one hot encoding: a process of converting categorical variables into a binary vector representation.

vector: a one-dimensional array of numbers
matrix: a two-dimensional array of numbers
tensor: a multi-dimensional array of numbers

class refers to a category or label: in this case, the category is a character(a-z & .)
'''
import torch.nn.functional as F
xs_hot_encoded = F.one_hot(xs_tensor, num_classes=27).float()
xs_hot_encoded
# %%
'''
the shape of the one hot encoded tensor is (sequence_length, num_classes)
xs_tensor has 5 elements: tensor([ 0,  5, 13, 13,  1]) -> 1D tensor

num_classes is 27 because there are 27 characters (a-z & .) 

xs_hot_encoded.shape is (5, 27) 2D tensor
each index would be converted into 27-element vector
[
 [1, 0, 0, ..., 0],  # One-hot for index 0
 [0, 0, 0, 0, 0, 1, ..., 0],  # One-hot for index 5
 [0, 0, 0, ..., 1, 0, 0],  # One-hot for index 13
 [0, 0, 0, ..., 1, 0, 0],  # One-hot for index 13
 [0, 1, 0, ..., 0]   # One-hot for index 1
]

'''
print(f"xs_hot_encoded.shape: {xs_hot_encoded.shape}")
print(f"xs_hot_encoded.dtype: {xs_hot_encoded.dtype}")
# %%
plt.imshow(xs_hot_encoded)

# %%
'''
weights_matrix: 27x27 matrix
xs_hot_encoded: 5x27 matrix

the dot product of xs_hot_encoded and weights_matrix is a 5x27 matrix
'''
g = torch.Generator().manual_seed(2147483647 + 1)
weights_matrix = torch.randn((27,27), generator=g)
weights_matrix

# %%
'''
(5, 27) @ (27, 27) = (5, 27)
each neuron = x1 * weight1 + x2 * weight2 + ... + x5 * weight5
x1 -> a row vector of 27 elements [1, 0, 0, ..., 0]
weight1 -> a column vector of 27 elements [w11, w12, ..., w1n]
x1 * weight1 = 1 * w11 + 0 * w12 + ... + 0 * w1n = w11
'''
xs_weights = (xs_hot_encoded @ weights_matrix)
xs_weights
# %%
xs_weights[3,13] # (xs_hot_encoded[3] * weights_matrix[:,13]).sum()

# %%
logits = xs_hot_encoded @ weights_matrix # log counts
# softmax
counts = logits.exp() # equivalent to N
probs = counts / counts.sum(1, keepdim=True) # normalize the counts
'''
xs: [0, 5, 13, 13, 1]
probs[0] is the probabilities of 27 elements 
and highest probability is the next character of xs[0] -> '.'
'''
probs[0]
# %%
nlls = torch.zeros(5)
for i in range(5):
      # i-th bigram
      x = xs_tensor[i].item() # input character index
      y = ys_tensor[i].item() # output character index
      print('-----------')
      print(f"bigram example {i+1}: {indexToChar[x]}{indexToChar[y]} (indexes {x},{y})")
      print('input to the neural net: ', x)
      print('output probabilities from the neural net: ', probs[i])
      print('label (actual next character): ', y)
      p = probs[i,y]
      print('probaility assigned by the net to the correct character: ', p.item())
      logp = torch.log(p)
      print('log likelihood: ', logp.item())
      nll = -logp
      print('negative log likelihood: ', nll.item())
      nlls[i] = nll

print("=========")
print('average negative log likelihood, i.e. loss = ', nlls.mean().item())

# %%
# ----------- 1st OPTIMIZATION !!! ----------------
# %%
xs_tensor
# %% 
ys_tensor

# %%
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# %%
# Forward pass
xenc = F.one_hot(xs_tensor, num_classes=27).float() # input to network: one-hot encoding
logits = xenc @ W # prefict log-counts
counts = logits.exp() # counts
probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
# %%
(probs[0,5], probs[1, 13], probs[2,13], probs[3,1], probs[4,0])
# %%
torch.arange(5)

# %%
loss = -probs[torch.arange(5), ys].log().mean()
loss
 # %%
# backward pass
W.grad = None # set to zero the gradient
loss.backward()

# %%
W.data += 0.1 * W.grad

# %%
# --------- !!!! 2nd OPTIMIZATION !!! yay ------------
xs, ys = [], []
for w in words:
      chs = ['.'] + list(w) + ['.']
      for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = charToIndex[ch1]
            ix2 = charToIndex[ch2]
            xs.append(ix1)
            ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# Initialize the network
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
# %%
# gradient descent
for k in range(10):

      # forward pass
      xenc = F.one_hot(xs, num_classes=27).float()
      logits = xenc @ W # predict log-counts
      counts = logits.exp() # counts
      probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
      # 0.01 * (W**2).mean() -> regulation
      loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
      print(loss.item())

      # backward pass
      W.grad = None
      loss.backward()

      # update 
      # 50 is the learning rate
      W.data += -50 * W.grad 

# %%
0.01 * (W**2).mean()
# %%
