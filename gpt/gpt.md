# Transformer

> Attention is what you need
> https://arxiv.org/html/1706.03762v7

## Hyperparameters

- **`batch_size`**: Number of sequences processed in parallel during training.
- **`block_size`**: Maximum context length for predictions, limiting how far back the model can look.
- **`max_iters`**: Total number of training iterations.
- **`eval_interval`**: Frequency of performance evaluation on training and validation sets.
- **`learning_rate`**: Step size for the optimizer, affecting the speed of learning.
- **`device`**: Determines whether to use GPU or CPU for computations.
- **`eval_iters`**: Number of iterations to average over when estimating loss.
- **`n_embd`**: Dimensionality of token embeddings and hidden states.
- **`n_head`**: Number of attention heads in the multi-head attention mechanism.
- **`n_layer`**: Number of transformer blocks stacked in the model.
- **`dropout`**: Dropout rate for regularization to prevent overfitting.

## Transformer Architecture Components

- **Embedding Layers**: Convert tokens and positions into dense vectors for processing.
- **Dense Vectors**: Compact, information-rich representations where most elements are non-zero, used to capture complex patterns and relationships in data.
- **Attention Mechanism**: Allows tokens to focus on different parts of the input sequence.
- **FeedForward Network**: Processes each token independently with non-linear transformations.
- **Residual Connections**: Facilitate gradient flow and learning of identity mappings.
- **Layer Normalization**: Stabilizes learning by normalizing inputs to each layer.
- **Output Layer**: Projects final hidden states to logits for next token prediction.

### Embedding Layer

- **Dense Vectors**:

  - Continuous values
  - Compact, information-rich representations where most elements are non-zero
    - By contrast to sparse vectors, which may have many 0 elements
  - Used in Embeddings
    - dense vectors are used in the token and position embeddings.
    - embeddings map discrete tokens and their positions into continuous vector spaces, allowing the model to process and learn from the input data effectively

- **Token Embeddings**:

  - Convert discrete tokens (words or characters) into dense vectors of size `n_embd`.
  - Implemented using `nn.Embedding`, which acts as a lookup table mapping each token index to a vector.
  - Example: `self.token_embedding_table = nn.Embedding(vocab_size, n_embd)`

- **Position Embeddings**:

  - Encode the position of each token in the sequence to provide the model with information about the order of tokens.
  - Also implemented using `nn.Embedding`, mapping each position index to a vector.
  - Example: `self.position_embedding_table = nn.Embedding(block_size, n_embd)`

- **Combined Embeddings**:
  - The sum of token and position embeddings provides the input to the transformer blocks, allowing the model to understand both the content and order of the sequence.
  - Example: `x = tok_emb + pos_emb`

---

## Self-Attention

self-attention is a mechanism that allows each token in a sequence to attend to all other tokens

### What is Attention?

Attention is a mechanism that allows models to focus on specific parts of the input sequence when making predictions. It can be thought of as a way for the model to "attend" to relevant information, much like how humans focus on certain parts of a scene or text when processing information.

### Lower Triangular Matrix

```
wei = torch.tril(torch.ones(T,T))
```

To maintain the causal structure of language (i.e., predicting the next word based on previous words), we use a lower triangular matrix as a mask

#### Why use a Lower Triangular Matrix?

1. causal masking: in self-attention, each toekn should only attend only attend to itself and the tokens before it, not the ones after. (during training, we want to predict the next token without "cheating" by looking at future tokens)

2. matrix mulitplication: when computing attention scores, a lower triangular matrix ensures that the attention mechanism only considers the current and past tokens. this is done by setting the upper triangular part of the attention score matrix to 0, effectively ignoring future tokens

3. this approach respects the natural order of the sequence, which is crucial for tasks like language modeling where the order of words matters

#### Math example: masking with a lower triangular matrix

```
sequence = [
   2,
   4,
   6
]

lower_triangular_matrix = [
   [1,0,0],
   [1,1,0],
   [1,1,1]
]

masking with matrix multiplication
result = sequence @ lower_triangular_matrix
result = [
   [2],
   [2+4],
   [2+4+6]
] = [
   2, only influenced by itself (2)
   6, is the sume of the first 2 elements (2+4)
   12, the sum of all three elements (2+4+6)
]
```

### attention-weighted sum (`xbow`)

#### Code examples

- below 2 code snippets are designed to produce the same result (attention-weighted sum)

```python
#  version 1
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t,C)
        '''
        xbow[b,t] is the average of the sequence of feature vectors from start up to the `t` for b-th batch

        xprev.shape -> (t,C)

        mean(xprev, 0) -> 0 is the dimension -> refers to the time steps -> computes the mean across all time steps for each feature
        '''
        xbow[b,t] = torch.mean(xprev, 0)
```

```python
# version 2
wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B,T,T) @ (B,T,C) -----> (B,T,C)
```

xbow refers to **attention-weighted sum** of embeddings in the context of self-attention mechanisms.

1. self-attention mechanism: in self-attention, each toekn in a sequence computes attention scores with every other token. these scores determine how much influence each token should have on the others

2. attention-weighted sum: `xbow` is computed by taking a weighted sum of the token embeddings, where the weights are the attention scores. this results in a new representation for each token that incorporates information from other tokens in the sequence.

### Conclusion about `xbow` & Lower triangular

`xbow` provides a contextualized representation of each token, incorporating information from the sequence while respecting causal dependecies. The **lower triangular matrix** ensures that each token's xbow is influenced only by itself and preceding tokens, maintaining the sequence's natural order and preventing information leakage from future tokens
**This is essential for tasks like languar modeling, where predicting the next token based on the past context is crucial.**

### Key, Query, & Value in Self-Attention

- Query (Q): Represents the current token's perspective, used to compute attention scores with other tokens.
- Key (K): Represents each token's identity, used to compare against the query to determine relevance.
- Value (V): Contains the actual information to be aggregated, weighted by attention scores.

#### Mathematical Example

**Consider a simple sequence: "The cat sat", with each word represented by a 2D vector:**

```
"The" = [1, 0]
"cat" = [0, 1]
"sat" = [1, 1]
```

**Step 1: Compute Key, Query, and Value Vectors**

```
Key vectors: K1 = [1, 0], K2 = [0, 1], K3 = [1, 1]
Query vectors: Q1 = [1, 0], Q2 = [0, 1], Q3 = [1, 1]
Value vectors: V1 = [1, 0], V2 = [0, 1], V3 = [1, 1]
```

**Step 2: Compute Attention Scores**
For each word, compute the dot product of the query with each key:

\[ \text{Score}(Q_i, K_j) = Q_i \cdot K_j \]

```
For "The" (Q1 = [1, 0]):
Score with "The" (K1 = [1, 0]):
Score(Q1, K1) = [1, 0] · [1, 0] = 1 * 1 + 0 * 0 = 1

Score with "cat" (K2 = [0, 1]):
Score(Q1, K2) = [1, 0] · [0, 1] = 1 * 0 + 0 * 1 = 0

Score with "sat" (K3 = [1, 1]):
Score(Q1, K3) = [1, 0] · [1, 1] = 1 * 1 + 0 * 1 = 1
---------------------------------------------

For "cat" (Q2 = [0, 1]):
Score with "The" (K1 = [1, 0]):
Score(Q2, K1) = [0, 1] · [1, 0] = 0 * 1 + 1 * 0 = 0

Score with "cat" (K2 = [0, 1]):
Score(Q2, K2) = [0, 1] · [0, 1] = 0 * 0 + 1 * 1 = 1

Score with "sat" (K3 = [1, 1]):
Score(Q2, K3) = [0, 1] · [1, 1] = 0 * 1 + 1 * 1 = 1
---------------------------------------------

For "sat" (Q3 = [1, 1]):
Score with "The" (K1 = [1, 0]):
Score(Q3, K1) = [1, 1] · [1, 0] = 1 * 1 + 1 * 0 = 1

Score with "cat" (K2 = [0, 1]):
Score(Q3, K2) = [1, 1] · [0, 1] = 1 * 0 + 1 * 1 = 1

Score with "sat" (K3 = [1, 1]):
Score(Q3, K3) = [1, 1] · [1, 1] = 1 * 1 + 1 * 1 = 2
```

**Step 3: Normalize Attention Scores**

Apply softmax to convert scores into probabilities:

- For "The": Softmax([1, 0, 1]) = [0.422, 0.156, 0.422]

**Step 4: Compute Weighted Sum of Values**

Use normalized scores to weight the value vectors. The value vectors for the sequence "The cat sat" are:

- Value vectors: V1 = [1, 0], V2 = [0, 1], V3 = [1, 1]

For the word "The", the normalized attention scores are [0.422, 0.156, 0.422]. We use these scores to compute the weighted sum of the value vectors:

```
Output for "The":
0.422 * V1 + 0.156 * V2 + 0.422 * V3
= 0.422 * [1, 0] + 0.156 * [0, 1] + 0.422 * [1, 1]
= [0.422, 0] + [0, 0.156] + [0.422, 0.422]
= [0.844, 0.578]
```

#### Conclusion

The self-attention mechanism uses key, query, and value vectors to compute attention scores, which are normalized and used to produce a context-aware representation for each token. This process allows the model to focus on relevant parts of the sequence, crucial for tasks like language modeling.

### Encoder vs Decoder Attention

#### Encoder Attention

- In encoder block, all tokens can communicate freely, as there is no masking.
  - allows the model to capture the full context of the input sequence
- Use Case: Encoder attention is used in tasks where understanding the entire input context is crucial, such as in machine translation or text classification

```python
torch.manual_seed(1337)
B,T,C = 4,8,32
x = torch.randn(B,T,C)

# define the head size for the attention mechanism
head_size = 16

# linear transformation for keys, queires, and values
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
vlaue = nn.Linear(C, head_size, bias=False)

# Compute the key, query, and value vectors
k = key(x) # (B,T,16)
q = query(x) # (B,T,16)
v = value(x) # (B,T,16)

# Compute the attention scores
wei = q @ k.transpose(-2,-1) # (B,T,16) @ (B,16,T) -> (B,T,T)

# Apply softmax to get the attention weights
wei = F.softmax(wei, dim=-1)

# Compute the output as the wei
out = wei @ v # (B,T,T) @ (B,T,16) -> (B,T,16)
```

- **No Masking:** The absence of masking allows each token to attend to every other token, capturing the full context.
- **Attention Weights:** Calculated using the dot product of queries and keys, normalized with softmax.
- **Output:** A context-aware representation of each token, useful for encoding the input sequence.

#### Decoder Attention

- In a decoder block, triangular masking is applied to prevent future tokens from inflluencing the current token.
- Used in tasks where the model needs to generate sequences such as text generation

```
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

head_size = 16

# Linear transformations for keys, queries, and values
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# Compute the key, query, and value vectors
k = key(x)  # (B, T, 16)
q = query(x)  # (B, T, 16)
v = value(x)  # (B, T, 16)

# Compute the attention scores
wei = q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) -> (B, T, T)

# Apply triangular masking to prevent future token influence
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))

# Apply softmax to get the attention weights
wei = F.softmax(wei, dim=-1)

# Compute the output as the weighted sum of the value vectors
out = wei @ v  # (B, T, T) @ (B, T, 16) -> (B, T, 16)
```

- **Triangular Masking:** Ensures that each token can only attend to itself and previous tokens, maintaining the causal structure.
- **Attention Weights:** Calculated similarly to encoder attention but with masking to prevent future token influence.
- **Output:** A context-aware representation that respects the sequence order, essential for generating sequences.

#### Summary Encoder vs Decoder Attention

- **Encoder Attention:** Allows full token interaction without masking, capturing the entire input context.
- **Decoder Attention:** Uses triangular masking to maintain sequence order, crucial for autoregressive tasks.

### Scaled Attention

Scaled attention refers to a modification applied to the attention scores before they are passed through the `softmax`

- scaling is crucial for maintaining numerical stability and ensuring effective learning
- scaling factor is `1/sqrt(head_size)`

```python
wei = q @ k.transpose(-2,-1) * k.shape[-1] **-0.5
```

- `k.shape[-1] **-0.5` -> scaling

#### What is Softmax?

- **Softmax Function**:
  - Converts attention scores into a probability distribution, ensuring the weights sum to one.
  - Formula: \(\text{softmax}(z*i) = \frac{e^{z_i}}{\sum*{j} e^{z_j}}\)
  - Example in code: `wei = F.softmax(wei, dim=-1)`

## Cross-Attention

Cross-attention is a mechanism used in transformer models, particularly in the decoder, to incorporate information from an external source, typically the encoder's output.

- **Queries (Q):** Derived from the decoder's current state, representing the current position in the output sequence being generated.
- **Keys (K) and Values (V):** Derived from the encoder's output, which contains the encoded information of the input sequence.
- **Purpose:** Allows the decoder to focus on relevant parts of the input sequence while generating the output, leveraging the contextual information encoded by the encoder.
- **Process:**
  - Compute attention scores by taking the dot product of queries and keys.
  - Normalize these scores using softmax to produce attention weights.
  - Use these weights to compute a weighted sum of the value vectors, resulting in a context-aware representation for each position in the decoder's current state.

This mechanism enables the decoder to effectively generate sequences by focusing on the most relevant parts of the input, guided by the encoded representations from the encoder.

#### Code Example

```python
B, T_enc, T_dec, C = 4, 8, 6, 32  # batch, encoder time, decoder time, channels
encoder_output = torch.randn(B, T_enc, C)  # Encoder output
decoder_input = torch.randn(B, T_dec, C)  # Decoder input (current state)

# Define the head size for the attention mechanism
head_size = 16

# Linear transformations for keys, queries, and values
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# Compute the key and value vectors from the encoder output
k = key(encoder_output)  # (B, T_enc, 16)
v = value(encoder_output)  # (B, T_enc, 16)

# Compute the query vectors from the decoder input
q = query(decoder_input)  # (B, T_dec, 16)

# Compute the attention scores
wei = q @ k.transpose(-2, -1)  # (B, T_dec, 16) @ (B, 16, T_enc) -> (B, T_dec, T_enc)

# Apply softmax to get the attention weights
wei = F.softmax(wei, dim=-1)

# Compute the output as the weighted sum of the value vectors
out = wei @ v  # (B, T_dec, T_enc) @ (B, T_enc, 16) -> (B, T_dec, 16)

print(out.shape)
```

**1. Why the length of decoder and encoder are different?**

- Machine Translation: Consider translating "Hello world" (2 tokens) to "Bonjour le monde" (3 tokens). The encoder processes the 2-token input, while the decoder generates a 3-token output, resulting in different sequence lengths.

## Multi-Head Attention

Multi-head attention is a mechanism that enhances the model's ability to focus on different parts of the input sequence simultaneously. It achieves this by using multiple attention heads, each operating on a different subspace of the input.

### Key Concepts:

1. **Multiple Heads**:

   - Each head processes the same input data but in a different subspace, allowing the model to capture diverse patterns and dependencies.
   - The input is split into `n_head` smaller subspaces, each with a dimensionality of `head_size = n_embd // n_head`.

2. **Independent Processing**:

   - Each head independently computes attention scores using its own set of key, query, and value vectors.
   - This parallel processing enables the model to efficiently learn different aspects of the input.

3. **Concatenation and Projection**:

   - The outputs from all heads are concatenated along the last dimension, resulting in a tensor of shape `(B, T, head_size * n_head)`.
   - A linear layer then projects this concatenated output back to the original embedding size `(B, T, n_embd)`, combining the diverse information captured by each head into a single representation.

4. **Benefits**:
   - **Richer Representations**: By focusing on different parts of the input, multi-head attention provides a more nuanced representation.
   - **Flexibility**: Allows the model to learn various attention patterns, such as focusing on local context or long-range dependencies.
   - **Improved Performance**: Enhances the model's ability to capture complex dependencies, leading to better performance on tasks like language modeling and translation.

### Multi-Head Attention Process

#### Dimensionality Changes:

1. **Input to Multi-Head Attention**:

   - **Input Tensor `x`**: Shape `(B, T, C)`
     - `B`: Batch size
     - `T`: Sequence length (number of tokens)
     - `C`: Embedding size (original dimensionality of each token)

2. **Linear Transformations for Keys, Queries, and Values**:

   - Each head has its own linear layers to transform the input into keys, queries, and values.
   - **Key, Query, Value for Each Head**: Shape `(B, T, head_size)`
     - `head_size = C // n_head` (assuming `C` is divisible by `n_head`)

3. **Attention Scores Calculation**:

   - **Scores**: Shape `(B, T, T)`
     - Computed as the dot product of queries and keys, scaled by `1/sqrt(head_size)`.
     - Represents the attention scores for each token with respect to every other token.

4. **Attention Weights**:

   - **Weights**: Shape `(B, T, T)`
     - Obtained by applying softmax to the attention scores.

5. **Weighted Sum of Values**:
   - **Output for Each Head**: Shape `(B, T, head_size)`
     - Calculated as the weighted sum of the value vectors using the attention weights.

#### Concatenation and Projection:

1. **Concatenation of Head Outputs**:

   - **Concatenated Output**: Shape `(B, T, head_size * n_head)`
     - All head outputs are concatenated along the last dimension.

2. **Projection Back to Original Size**:
   - **Final Output**: Shape `(B, T, C)`
     - The concatenated output is projected back to the original embedding size using a linear layer.

#### Head Size and Number of Heads:

- **Head Size**: The dimensionality of the subspace that each attention head operates on. It is calculated as `head_size = C // n_head`.
- **Number of Heads (`n_head`)**: Determines how many parallel attention mechanisms are used. More heads allow the model to capture a wider range of patterns and dependencies.
- **Purpose**: Each head processes the entire sequence but with its own set of learned parameters, allowing it to focus on different aspects of the sequence. This enables the model to capture diverse relationships and dependencies without physically breaking the sequence into sections.

#### Example in Code:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

- **`self.heads`**: Each `Head` processes the input independently, producing an output of size `(B, T, head_size)`.
- **Concatenation**: The outputs from all heads are concatenated to form a tensor of shape `(B, T, head_size * n_head)`.
- **Projection (`self.proj`)**: A linear layer that projects the concatenated output back to the original embedding size `(B, T, C)`.

By leveraging multiple attention heads, the transformer model can effectively capture complex dependencies and relationships in the input data, enhancing its performance across various tasks.

## FeedForward

The feedforward network in a transformer model is a crucial component that processes each position of the sequence independently. It consists of two linear transformations with a non-linear activation function in between.

1. **Non-linearity**:

   - Introduces non-linear transformations, allowing the model to learn complex patterns and relationships.

2. **Position-wise Processing**:

   - the application of operations independently to each position (or token) in a sequence. In the context of a transformer model, this means that each token in the sequence is processed separately and identically by the feedforward network. This allows the model to enhance the representation of each token without considering its position relative to other tokens in the sequence.
   - Each token in the input sequence is processed independently using the same feedforward operation. This means that the transformation applied to one token does not affect the others.
   - This approach allows the model to enhance the representation of each token individually, which is crucial for capturing the unique characteristics of each token in the sequence.

3. **Increased Capacity**:
   - Temporarily expands the dimensionality to capture more complicated patterns.

### Process:

1. **Input**:

   - Tensor of shape `(B, T, C)`, where `B` is the batch size, `T` is the sequence length, and `C` is the embedding size.

2. **First Linear Transformation (Expansion)**:

   - Expands the dimensionality from `C` to `4 * C`.
   - **Output Shape**: `(B, T, 4 * C)`

3. **Non-linear Activation**:

   - Applies ReLU to introduce non-linearity.
   - **Output Shape**: `(B, T, 4 * C)`

4. **Second Linear Transformation (Reduction)**:

   - Projects back to the original dimensionality `C`.
   - **Output Shape**: `(B, T, C)`

5. **Dropout (Regularization)**:
   - Prevents overfitting by randomly setting a fraction of the units to zero during training.
   - **Output Shape**: `(B, T, C)`

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expansion
            nn.ReLU(),                      # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Reduction
            nn.Dropout(dropout)             # Regularization
        )

    def forward(self, x):
        return self.net(x)
```

## Residual Connections

- Residual Connection: In neural networks, a residual connection refers to the practice of adding the input of a layer to its output. This is done to create a shortcut path that allows gradients to flow more easily during backpropagation, helping to mitigate issues like the vanishing gradient problem.
- Purpose: The residual connection allows the network to learn modifications to the input rather than having to learn a complete transformation from scratch. This can make it easier for the network to learn identity mappings, which is particularly useful in very deep networks.

```python
class Block(nn.Module):
    ''' Transformer block: communication followed by computation '''

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connection for self-attention
        x = x + self.ffwd(self.ln2(x))  # Residual connection for feedforward
        return x
```

- **Layer Normalization**: Applied before the residual connection to stabilize learning.
- **Addition**: The input is added to the output of the layer, allowing the model to learn residual mappings.

## Note from Andrej Karpathy

1. Attention is a communication mechanism. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
2. There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
3. Each example across batch dimension is of course processed completely independently and never "talk" to each other
4. In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
5. "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
6. "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much. Illustration below
