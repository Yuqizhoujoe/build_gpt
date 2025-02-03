# WaveNet & deep convolutional neural network (CNN)

## Step Flattening

```
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    '''
    B: batch size
    T: Sequence length or time steps, which is the number of elements in a sequence.
    C: Number of channels or features, which is the number of features per element in the sequence.
    '''
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []

Flattening: (B, 8, 10) to (B, 4, 20)
Processing: Apply a linear layer or activation function
Flattening: (B, 4, 20) to (B, 2, 40)
Processing: Another transformation
Final Flattening: (B, 2, 40) to (B, 80)
```

The process of transforming a tensor from (B, 8, 10) to (B, 80) through intermediate steps like (B, 4, 20) and (B, 2, 40) is a structured way to progressively combine features from consecutive elements. Here's why this step-by-step flattening is often preferred over a direct flattening:
Reasons for Step-by-Step Flattening

1. Hierarchical Feature Learning:

- By flattening in stages, you allow the model to learn hierarchical features. Each stage can capture increasingly complex patterns by combining features from smaller groups of consecutive elements.
- For example, (B, 4, 20) combines every 2 elements, allowing the model to learn interactions between these pairs before further combining them.

2. Intermediate Processing:

- Between these flattening stages, you can insert additional processing layers (e.g., linear layers, batch normalization, activation functions) to transform and refine the features.
- This intermediate processing can enhance the model's ability to learn complex patterns by applying transformations at each stage.

3. Model Flexibility:

- This approach provides flexibility in model design. You can adjust the number of features or the type of processing applied at each stage, allowing for more tailored architectures.

4. Avoiding Information Loss:

- Directly flattening from (B, 8, 10) to (B, 80) might lead to a loss of temporal or sequential information. By flattening in stages, you preserve and gradually integrate this information.

5. Improved Gradient Flow:

- Intermediate layers can help in maintaining a good gradient flow during backpropagation, which is crucial for training deep networks effectively.
