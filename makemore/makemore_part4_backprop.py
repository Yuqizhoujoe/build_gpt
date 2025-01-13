# %% [markdown]
## Makemore part 4
# %%
import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt 
# %%
with open("names.txt", "r") as file:
    words = file.read().splitlines()
print(f"number of word: {len(words)}")
print(f"first 8 words: {words[:8]}")
# %%
