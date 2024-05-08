from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, gammas:Tensor, betas:Tensor):
    gammas = gammas.view(*(gammas.shape + (1,) * (x.dim() - gammas.dim())))
    betas = betas.view(*(betas.shape + (1,) * (x.dim() - betas.dim())))
    return (gammas * x) + betas

class FiLMGen(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(FiLMGen, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        betas, gammas = x.split(x.size(1) // 2, dim=1)
        return betas, gammas
