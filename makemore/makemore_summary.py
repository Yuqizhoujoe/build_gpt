# %%
import torch
import torch.nn.functional as F
# %%
xs = torch.tensor([0, 5, 13, 13, 1])
xs

# %%
ys = torch.tensor([5, 13, 13, 1, 0])
ys
# %%
# randomly initialiuze 27 neurons weights each neuron has 27 inputs
g = torch.Generator().manual_seed(2147483647)
weights = torch.randn((27, 27), generator=g)

# %%
# input to network: one-hot encoding 
x_one_hot_encoded = F.one_hot(xs, num_classes=27).float()

# %%


# predict log counts
logits = x_one_hot_encoded @ weights

# softmax
counts = logits.exp() # counts equivalent to N
# probabilities for next character
probs = counts / counts.sum(1, keepdims=True)

# %%
