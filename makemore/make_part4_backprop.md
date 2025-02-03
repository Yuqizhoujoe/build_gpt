# Manual Backpropation

## Fordward pass

### Activation Functions

**Common Activation Functions**

1. Sigmoid

- Formula: σ(x) = 1 / (1 + exp(-x))
- Range: (0, 1)
- Use: Often used in the output layer for binary classification.

2. Tanh

- Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
- Range: (-1, 1)
- Use: Common in hidden layers; zero-centered output.

3. ReLU (Rectified Linear Unit)

- Formula: f(x) = max(0, x)
- Range: [0, ∞)
- Use: Popular in hidden layers due to simplicity and efficiency.

4. Leaky ReLU

- Formula: f(x) = x if x > 0, else f(x) = α \* x (where α is a small constant)
- Range: (-∞, ∞)
- Use: Addresses the "dying ReLU" problem by allowing a small, non-zero gradient when the unit is not active.

5. Softmax

- Formula: σ(x_i) = exp(x_i) / sum(exp(x_j) for j in range(len(x)))
- Range: (0, 1), and the outputs sum to 1
- Use: Often used in the output layer for multi-class classification.

#### Purpose:

Non-linearity: Allows the network to learn complex patterns.
Gradient Flow: Helps in backpropagation by providing gradients for updating weights.
Output Transformation: Maps inputs to a desired range, often for interpretability or to meet the requirements of the next layer.

## Operations in Backpropation

### Why transform back to input shape?

1. Reduction opereations: Operations like `sum` and `max` reduce the dimensionality of the input tensor.
2. Gradient propagation: during backpropagation, you need to distribute the gradient of the reduced output back to the original input dimensions -> ensure that each element of the input tensor receives the correct portion of the gradient
3. Broadcasting: when you perform operations that involve broadcasting (like subtracting a scalar from a tensor), the gradient needs to be accumulated across the dimensions

### How to transform back to Input Shape

- Sum operation:
  if you sum over a dimension, the gradient of the sum w.r.t each element of the input
  you need to expand the gradient back to the original shape using operations like `torch.ones_like()` or by using `expand` or `repeat` to match the input shape

- Max operation:
  the gradient is non-zero only for the lements that were the maximum in the forward pass. You can use functions like `torch.nn.functional.one_hot` to create a mask that identifies the positions of the maximum values and then apply the gradient only to those positions

### Example `max()` & Broadcasting

- `max()`

```
logit_maxes = logits.max(1, keepdim=True).values
```

Use of one_hot in Backpropagation:

- Max Operation: In the forward pass, logits.max(1) finds the max value in each row.
- Gradient Propagation: During backpropagation, gradients should only affect the max elements.
- One-hot Encoding: F.one_hot creates a mask where only max positions are 1.
- Purpose: Ensures dlogit_maxes is applied only to max elements in dlogits.

```
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
```

- Broadcasting

```
norm_logits = logits - logit_maxes
```

- The use of sum(1) in the backward pass is to ensure that the gradient of the loss with respect to logit_maxes matches the shape of logit_maxes, which is (32, 1) in this case.

**Here's a brief explanation:**

- Forward Pass: The `max()` operation is applied along dimension 1 (across columns) to find the maximum value in each row of logits. This results in `logit_maxes` having a shape of (32, 1), where 32 is the batch size.
- Backward Pass: When computing the gradient of the loss with respect to `logit_maxes`, you need to accumulate the gradients from all elements in each row of `dnorm_logits` that were affected by the subtraction of `logit_maxes`. This is done using `sum(1)`, which sums across the columns, resulting in a gradient tensor that matches the shape of `logit_maxes` (i.e., (32, 1)).

This ensures that the gradient is correctly propagated back through the network, maintaining the necessary shape for further backpropagation steps.

```
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
```

### Broadcasting in Backpropagation

- Forward pass:

```python
bnraw = bndiff * bnvar_inv # (x_i - μ) / sqrt(σ^2 + ε)
```

#### What is Broadcasting ?

- When multiplying `bnvar_inv` (shape `(1, 64)`) with `bndiff` (shape `(batch_size, 64)`), broadcasting allows this operation to apply `bnvar_inv` to each row of `bndiff`.
- Broadcasting means that the single row of `bnvar_inv` is applied to each of the 32 rows of `bndiff`.
- This results in a tensor of the same shape as `bndiff`, i.e., `(batch_size, 64)`.

#### Backward pass: broadcasting

```
dbndiff = bnvar_inv * dbnraw
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
```

In the backward pass, the summation over the batch dimension ensures that the gradient reflects the cumulative effect of `bnvar_inv` across the entire batch.

#### Additional: Basic Example of Broadcasting

- Consider two arrays: `A` with shape `(3, 1)` and `B` with shape `(1, 4)`.
  - `A = [[1], [2], [3]]`
  - `B = [[4, 5, 6, 7]]`
- When you add `A` and `B`, broadcasting expands `A` to a shape of `(3, 4)` and `B` to a shape of `(3, 4)`:
  - `A` becomes `[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]`
  - `B` becomes `[[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]]`
- The result of `A + B` is:
  - `[[5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10]]`

### About keepdim()

#### Bias Addition (`h = W1 @ embcat + b1`):

- b1 as a Vector: In the forward pass, b1 is a 1D vector (e.g., shape (output_dim,)) that is broadcasted across the batch dimension when added to the result of W1 @ embcat. This broadcasting is straightforward because b1 is inherently a 1D vector, and the operation naturally extends it across the batch.

```
db1 = dprebn.sum(0)
```

- No `keepdim` needed: When computing the gradient of the bias, you sum over the batch dimension to get a vector of the same shape as b1. Since b1 is a vector, there's no need to maintain the batch dimension, so keepdim is not necessary.

#### Element-wise multiplication (`bnraw = bndiff * bnvar_inv`)

- bnvar_inv as a 2D Tensor: In this operation, bnvar_inv is a 2D tensor (e.g., shape (1, features)) that is broadcasted across the batch dimension to match bndiff (e.g., shape (batch_size, features)).

```
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
```

- Need for keepdim=True: When computing the gradient of bnvar_inv, you sum over the batch dimension. Using keepdim=True ensures that the result retains the 2D shape (1, features), which is necessary for any subsequent operations that expect a 2D tensor. This is important for maintaining compatibility with the original shape of bnvar_inv and for any broadcasting that might occur later.

### Cross-Entropy Loss and Softmax

```
dlogits = F.softmax(logits, 1)
```

1. Cross-Entropy Loss: In PyTorch, the `F.cross_entropy` function combines both the softmax activation and the negative log-likelihood loss in one efficient step. This is why you don't see an explicit softmax in the forward pass when using `F.cross_entropy`.
2. Softmax in Backward Pass: When you compute the gradient of the loss with respect to the logits, you need to consider the effect of the softmax function. The softmax function is applied to the logits to convert them into probabilities, and the cross-entropy loss is computed based on these probabilities.

#### Why `dim=1`?

- Dimension 1: When you apply softmax with dim=1, you are computing the softmax across the class dimension for each sample in the batch. This is necessary because the cross-entropy loss is computed for each sample based on its class probabilities.
- Gradient Calculation: In the backward pass, you need to compute the gradient of the loss with respect to the logits. This involves differentiating through the softmax function. By applying softmax with dim=1, you ensure that the gradient is computed correctly across the class probabilities for each sample.

### Takeaways

1. Use `sum(1)` (or the appropriate dimension) in the backward pass when you need to accumulate gradients across a dimension that was reduced in the forward pass.
2.

## Type of Operations

### Element-wise Operations

Definition: Operations that apply a function to each element of a tensor independently.
Examples:

- Addition: Adding two tensors element by element.

```
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = a + b  # Result: tensor([5, 7, 9])
```

- Multiplication: Multiplying two tensors element by element.

```
    c = a * b  # Result: tensor([4, 10, 18])
```

- Functions: Applying functions like torch.exp, torch.sin, etc., to each element.

```
    d = torch.exp(a)  # Result: tensor([2.7183, 7.3891, 20.0855])
```

### Other Types of Operations

1. Matrix Operations:
   Matrix Multiplication: Unlike element-wise multiplication, matrix multiplication involves a dot product between rows and columns.

```
     A = torch.tensor([[1, 2], [3, 4]])
     B = torch.tensor([[5, 6], [7, 8]])
     C = A @ B  # Result: tensor([[19, 22], [43, 50]])
```

2. Reduction Operations:
   Operations that reduce the dimensions of a tensor by aggregating values.

Examples:
Sum: torch.sum(a) reduces all elements to a single sum.
Mean: torch.mean(a) calculates the average of all elements.
Max: torch.max(a) finds the maximum value.

3. Broadcasting:
   A technique that allows operations on tensors of different shapes by automatically expanding the smaller tensor to match the shape of the larger one.
   Example:

```
     a = torch.tensor([1, 2, 3])
     b = torch.tensor([[1], [2], [3]])
     c = a + b  # Result: tensor([[2, 3, 4], [3, 4, 5], [4, 5, 6]])
```

4. Indexing and Slicing:
   Operations that involve selecting specific elements or sub-tensors.
   Example:

```
     a = torch.tensor([[1, 2, 3], [4, 5, 6]])
     b = a[0, :]  # Result: tensor([1, 2, 3])
```

5. Reshaping:
   Changing the shape of a tensor without changing its data.
   Example:

```
a = torch.tensor([1, 2, 3, 4])
b = a.view(2, 2)  # Result: tensor([[1, 2], [3, 4]])

a = torch.tensor([1, 2, 3, 4, 5, 6])
b = a.view(2, 3)  # Reshape to 2 rows and 3 columns
print(b)
# Output:
# tensor([[1, 2, 3],
#         [4, 5, 6]])
```

## PyTorch

### sum()

#### sum(0)

when you use `.sum(0)` in PyTorch, it sums over the rows, effectively reducing the first dimension (dimension 0) of the tensor. Here's a bit more detail:
Dimension 0: Refers to the rows of a 2D tensor. When you sum over dimension 0, you are summing down the columns, collapsing the rows into a single row.
For example, consider a tensor:

```
tensor = torch.tensor([
    [a, b, c],
    [d, e, f],
    [g, h, i]
])
```

Using `tensor.sum(0)` would result in:

```
# Resulting in a single row
tensor([a+d+g, b+e+h, c+f+i])
```

#### sum(1)

When you use .sum(1) in PyTorch, it sums over the columns, effectively reducing the second dimension (dimension 1) of the tensor. Here's an example to illustrate this:
Suppose you have a 2D tensor:

```
tensor = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
```

Using tensor.sum(1) would result in:

```
# Resulting in a single column
tensor([ 6, 15, 24])
```

### Understanding Scalar, Vector, and Tensor

1. Scalar

A scalar is a single number (a zero-dimensional entity). It has magnitude but no direction.

Example: `x = 5`
In PyTorch: `torch.tensor(5)`

2. Vector

A vector is a one-dimensional array of numbers, meaning it has both magnitude and direction.

Example: `[1, 2, 3]`
In PyTorch: `torch.tensor([1, 2, 3])`

3. Matrix

A matrix is a two-dimensional array of numbers arranged in rows and columns.
In PyTorch: `torch.tensor([[1, 2, 3], [4, 5, 6]])`

4. Tensor

A tensor is a generalization of scalars, vectors, and matrices to higher dimensions (n-dimensional array).
`torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])`

Tensors can be 3D, 4D, or more, making them crucial for representing data in deep learning, such as:

- Images (e.g., RGB images: [batch_size, height, width, channels])
- Videos (e.g., [batch_size, time_steps, height, width, channels])
- Text sequences (e.g., [batch_size, sequence_length, embedding_dim])
