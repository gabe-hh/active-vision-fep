import torch
from torch import Tensor

class SOM():
    def __init__(self, ref_vectors, gamma=10):
        self.ref_vectors = ref_vectors
        self.gamma = gamma

    def get_activations(self, x:Tensor):
        # Expand ref_vectors to match the batch size of x
        ref_vectors_expanded = self.ref_vectors.unsqueeze(0).expand(x.size(0), -1, -1)

        # Compute the norm for each pair of vectors in the batch
        diff = ref_vectors_expanded - x.unsqueeze(1)  # Align x along the second dimension
        norm = torch.exp(-self.gamma * torch.linalg.vector_norm(diff, dim=2) ** 2)
        
        # Compute the sum along the reference vector dimension
        sum_norm = torch.sum(norm, dim=1, keepdim=True)

        # Normalize the activations for each vector in the batch
        activations = norm / sum_norm

        return activations

    def size(self):
        return self.ref_vectors.shape[0]
    