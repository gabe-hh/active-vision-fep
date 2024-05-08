import random, math
import torch
from model.som import SOM
from model.model import VAE
from model.loss import FreeEnergyLoss
import torch.distributions as D

class VAEController():
    def __init__(self, model:VAE, x_views, y_views, goal_img, efe_fn=FreeEnergyLoss(), device='cpu', temperature=0.5, som:SOM=None, goal_precision = 0.65):
        self.model = model
        #self.viewpoints = torch.tensor(viewpoints).to(device)
        self.x_views = x_views
        self.y_views = y_views
        self.efe_fn = efe_fn
        if goal_img is None:
            self.no_goal = True
        else:
            self.no_goal = False
        self.goal_img = goal_img
        self.v_dist = D.Categorical(torch.ones(x_views.size*y_views.size).to(device))
        self.device = device
        self.temperature = temperature
        self.som=som
        self.goal_precision = goal_precision

    def plan(self, o, v, num_samples=1):
        with torch.no_grad():
            batch_size, c, h, w = o.size()
            latent_D, z = self.model.encode(o, v)
            self.efe_fn.prior = latent_D
            view_efes = torch.zeros(size=(batch_size,self.x_views.size, self.y_views.size))
            O_hat = torch.zeros(size=(batch_size,self.x_views.size, self.y_views.size, c, h, w))
            for i, x in enumerate(self.x_views):
                for j, y in enumerate(self.y_views):
                    v_next = torch.tensor([x,y]).repeat(batch_size,1).to(torch.float32).to(self.device)
                    if self.som is not None:
                        v_next = self.som.get_activations(v_next).to(self.device)
                    o_hat = self.model.decode(z, v_next)
                    D_next,_ = self.model.encode(o_hat, v_next, imagine=True)
                    epi = self.efe_fn.regularization_term(D_next, keep_batch_dim=True)
                    if self.no_goal:
                        ins = 0
                    else:
                        ins = self.efe_fn.log_likelihood(o_hat, self.goal_img) # Negative log-likelihood     
                    efe = ins - epi # This gives EFE but we want to minimise it, so negate it later
                    view_efes[:,i,j] = efe
                    O_hat[:,i,j,:,:,:] = o_hat.cpu()
            efes_ngtv = view_efes * -1
            flat_efes = efes_ngtv.clone().detach().flatten(start_dim=1)
            view_probs = torch.softmax(flat_efes / self.temperature, dim=1)
            self.v_dist = D.Categorical(view_probs)
            return view_efes, view_probs.view(view_efes.shape), O_hat

    def plan_no_obs(self, z, size = (1, 3, 32, 32)):
        batch_size, c, h, w = size
        latent_D = self.model.latent_distribution
        with torch.no_grad():
            self.efe_fn.prior = latent_D
            view_efes = torch.zeros(size=(batch_size,self.x_views.size, self.y_views.size))
            view_epist = torch.zeros(size=(batch_size,self.x_views.size, self.y_views.size))
            view_instr = torch.zeros(size=(batch_size,self.x_views.size, self.y_views.size))
            O_hat = torch.zeros(size=(batch_size,self.x_views.size, self.y_views.size, c, h, w))
            for i, x in enumerate(self.x_views):
                for j, y in enumerate(self.y_views):
                    v_next = torch.tensor([x,y]).repeat(batch_size,1).to(torch.float32).to(self.device)
                    if self.som is not None:
                        v_next = self.som.get_activations(v_next).to(self.device)
                    o_hat = self.model.decode(z, v_next)
                    D_next,_ = self.model.encode(o_hat, v_next, imagine=True)
                    epi = self.efe_fn.regularization_term(D_next, keep_batch_dim=True)
                    if self.no_goal:
                        ins = 0
                    else:
                        ins = self.efe_fn.log_likelihood(o_hat, self.goal_img, variance=1/self.goal_precision) # Negative log-likelihood     
                    efe = ins - epi # This gives EFE but we want to minimise it, so negate it later
                    view_efes[:,i,j] = efe
                    view_epist[:,i,j] = epi
                    view_instr[:,i,j] = ins
                    O_hat[:,i,j,:,:,:] = o_hat.cpu()
            efes_ngtv = view_efes * -1
            flat_efes = efes_ngtv.clone().detach().flatten(start_dim=1)
            view_probs = torch.softmax(flat_efes / self.temperature, dim=1)
            self.v_dist = D.Categorical(view_probs)
            return view_efes, view_probs.view(view_efes.shape), O_hat, view_epist, view_instr

    def imagine_obs(self, v):
        z = self.model.latent_distribution.rsample()
        return self.model.decode(z, v)

    def percieve(self, o, v):
        with torch.no_grad():
            return self.model.encode(o,v)

    def get_random_view(self):
        i = random.randint(0, self.x_views.size-1)
        j = random.randint(0, self.y_views.size-1)
        return torch.tensor([self.x_views[i], self.y_views[j]], 
                            dtype=torch.float32, device=self.device).unsqueeze(0)

    # def get_next_view(self):
    #     v_index = self.v_dist.sample()
    #     print(f'New index: {v_index}')
    #     y_size = torch.tensor(self.y_views.size, dtype=v_index.dtype, device=v_index.device)
    #     i = v_index // y_size
    #     j = v_index % y_size
    #     print(f'i, j : {i}, {j}')
    #     return torch.tensor([self.x_views[i], self.y_views[j]], 
    #                         dtype=torch.float32, device=self.device).unsqueeze(0)

    def get_view_from_index(self, indices):
        # Retrieve the x and y view coordinates using the indices
        x_view = self.x_views[indices[0]]
        y_view = self.y_views[indices[1]]
        
        return torch.tensor([x_view, y_view], 
                            dtype=torch.float32, device=self.device).unsqueeze(0)

    def get_next_view(self):
        # Sample an action from the distribution
        sampled_index = self.v_dist.sample().item()
        
        num_y_views = len(self.y_views)
        
        # Correct calculation of the 2D indices from the flattened index
        x_index = sampled_index // num_y_views  # Divide by num_y_views instead of num_x_views
        y_index = sampled_index % num_y_views  # Use modulo with num_y_views
        
        # Retrieve the x and y view coordinates using the indices
        x_view = self.x_views[x_index]
        y_view = self.y_views[y_index]
        
        return torch.tensor([x_view, y_view], 
                            dtype=torch.float32, device=self.device).unsqueeze(0), (x_index, y_index)
