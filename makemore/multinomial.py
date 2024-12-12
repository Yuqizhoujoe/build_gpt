# %%
import torch 

g = torch.Generator().manual_seed(2147483647)
# Generate 10 random numbers between 0 and 1 [0, 1)
rand_numbers = torch.rand(10, generator=g)
probabilities = rand_numbers / rand_numbers.sum() # normalize
print(probabilities)
# %%
