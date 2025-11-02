# %% [markdown]
"""
## Build a GPT
"""
# %%
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)
n_embd = 32
print("".join(chars))
print(vocab_size)
# %%
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))
# %%
# let's encode the entire text dataset and store it into a tensor
import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])
# %%
# let's now split up the data into train and validation sets
n = int(0.9 * len(data))
train_data = data[n:]
val_data = data[:n]
# %%
block_size = 8
x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
# %%
torch.manual_seed(1337)
batch_size = 4  # how many independent sequences will be process in parallel?
block_size = 8  # what is the maximum context length for predictions?


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indices])
    return x, y


xb, yb = get_batch("train")

"""
xb: 
[[61,  6,  1, 52, 53, 58,  1, 58],
[56,  6,  1, 54, 50, 39, 52, 58],
[58,  1, 58, 46, 47, 57,  1, 50],
[10,  0, 32, 46, 43, 56, 43,  1]]

yb:
[[ 6,  1, 52, 53, 58,  1, 58, 47],
[ 6,  1, 54, 50, 39, 52, 58, 43],
[ 1, 58, 46, 47, 57,  1, 50, 47],
[ 0, 32, 46, 43, 56, 43,  1, 42]]

[61] -> 6
[61, 6] -> 1
[61, 6, 1] -> 52
[61, 6, 1, 52] -> 53
[61, 6, 1, 52, 53] -> 58
[61, 6, 1, 52, 53, 58] -> 1
[61, 6, 1, 52, 53, 58, 1] -> 58
[61, 6, 1, 52, 53, 58, 1, 58] -> 47
...
"""
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")
# %%
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(
    decode(
        m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[
            0
        ].tolist()
    )
)

# %%
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
# %%
batch_size = 32
for steps in range(10000):

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
# %%
print(
    decode(
        m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=300)[
            0
        ].tolist()
    )
)
# %% [markdown]
## The mathematicalk trick in self-attention
# %%
import torch

torch.manual_seed(1337)
B, T, C = 4, 8, 2  # batch, time, channels
x = torch.randn(B, T, C)
x.shape
# %%
"""
version 1
xbow: attention-weighted sum 
"""
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]  # (t,C)
        """
        xbow[b,t] is the average of the sequence of feature vectors from start up to the `t` for b-th batch
        
        xprev.shape -> (t,C) 
        
        mean(xprev, 0) -> 0 is the dimension -> refers to the time steps -> computes the mean across all time steps for each feature
        """
        xbow[b, t] = torch.mean(xprev, 0)
# %%
a = torch.tril(torch.ones(3, 3))
print(a)
a = a / torch.sum(a, 1, keepdim=True)
print(a)
b = torch.randint(0, 10, (3, 2)).float()
print(b)
c = a @ b
print(c)
# %%
"""
version 2
wei: weights
"""
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x  # (B,T,T) @ (B,T,C) -----> (B,T,C)
torch.allclose(xbow, xbow2)  # xbow2 == xbow
# %%
"""
version 3
"""
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)
# %%
"""
version 4: self-attention
"""
torch.manual_seed(1337)
B, T, C = 4, 8, 32  # batch, time, channels
x = torch.randn(B, T, C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B,T,16)
q = query(x)  # (B,T,16)
wei = q @ k.transpose(-2, -1)  # (B,T,16) @ (B,16,T) -> (B,T,T)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v

out.shape
# %%
