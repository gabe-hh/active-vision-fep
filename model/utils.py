import numpy as np
import torch
import torch.nn.functional as F

def calculate_padding(input_size, output_size, kernel_size, stride):
    # Formula to calculate the required padding
    return max((output_size - 1) * stride + kernel_size - input_size, 0) // 2

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

def gaussian_nll(mu, log_sigma, x):
    term1 = 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
    term2 = (log_sigma + 0.5 * np.log(2 * np.pi))
    print(f"term 1: {term1.sum()}")
    print(f"term1 shape: {term1.shape}")
    print(f"term 2: {term2.sum()}")
    print(f"term2 shape: {term2.shape}")
    print(f"sum: {(term1+term2).sum()}")
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + (log_sigma + 0.5 * np.log(2 * np.pi))

def sample_gumbel(shape, eps=1e-20, device='cuda'):
    U = torch.rand(shape)
    if device=='cuda':
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, latent_size, num_cats, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y.view(-1, latent_size * num_cats)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_size * num_cats)
