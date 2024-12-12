import torch

def log_tensor(tensor: torch.Tensor):
    rows, cols = tensor.shape
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(round(tensor[i, j].item(), 2))
        matrix.append(row)
    
    for row in matrix:
        print(row)
