import torch
import torch.distributions as D
import torch.nn as nn

class FreeEnergyLoss(nn.Module):
    def __init__(self, prior_args=(0, 1), mse_reduction_mode='sum'):
        super(FreeEnergyLoss, self).__init__()
        self.prior = D.Normal(*prior_args)
        self.mse_reduction_mode = mse_reduction_mode

    def set_prior(self, prior:D.Distribution):
        self.prior = prior

    def reconstruction_term(self, o_hat, o, var=1, keep_batch_dim=False):
        # The reconstruction term computes the summed MSE between reconstructed 
        if keep_batch_dim:
            return torch.nn.functional.mse_loss(o_hat, o, reduction=self.mse_reduction_mode)/(2*var)
        else:
            return torch.mean(torch.nn.functional.mse_loss(o_hat, o, reduction=self.mse_reduction_mode)/(2*var))
        return torch.sum((o_hat[0:i,:,:,:]-o[0:i,:,:,:])**2)

        mse = F.mse_loss(o_hat, o, reduction='mean')
        log_sigma_opt = 0.5 * mse.log()
        r_loss = 0.5 * torch.pow((o_hat - o) / log_sigma_opt.exp(), 2) + log_sigma_opt
        rec = r_loss.sum()

        # log_sigma = ((o - o_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()
        # log_sigma = softclip(log_sigma, -6)
        # rec = gaussian_nll(o_hat, log_sigma, o).sum()
        # print(f"Reconstruction Loss: {rec.item()}")
        return rec

    def regularization_term(self, posterior, keep_batch_dim=False):
        #return D.kl_divergence(posterior, self.prior).sum()
        if keep_batch_dim:
            return D.kl_divergence(posterior, self.prior).sum(dim=1)
        else:
            return torch.mean(D.kl_divergence(posterior, self.prior).sum(dim=1))
    
    def log_likelihood(self, o, goal, variance=0.65): # Compute the homoscedastic log-likelihood
        variance = torch.tensor(variance, dtype=torch.float32)
        diff = (o - goal) ** 2
        log_likelihood = 0.5 * (diff / variance + torch.log(2 * torch.pi * variance))
        return log_likelihood.sum(dim=[1, 2, 3]).mean(dim=0)  # Sum over all dimensions except the batch dimension

    def forward(self, o, o_hat, posterior, var=1):
        reg = self.regularization_term(posterior)
        rec = self.reconstruction_term(o_hat, o, var)
        return reg, rec
