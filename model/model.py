from torch import Tensor
import torch
import torch.nn as nn
import torch.distributions as D
from spatial_softmax import SpatialSoftmax, InverseSpatialSoftmax
from film import FiLM, FiLMGen
from utils import gumbel_softmax

class PosteriorModel(nn.Module):
    def __init__(self, latent_size, aux_size, conv_config, in_size=(32,32), filmgen_hidden_dim=[64], use_ssm=False, ssm_temp=0.5, fc_sizes = None):
        super(PosteriorModel, self).__init__()
        self.use_ssm = use_ssm
        # Initialize the FiLM generator
        if fc_sizes is not None:
            fc_total_size = sum(fc_sizes)
        else:
            fc_total_size = 0
        if filmgen_hidden_dim is not None:
            film_gen_output_size = sum([config['filters'] * 2 for config in conv_config if config.get('film', False)]) + (fc_total_size*2)
            self.film_gen = FiLMGen(aux_size, film_gen_output_size, filmgen_hidden_dim)
        else:
            print("No FiLM gen network parameters provided. If you have specified FiLM layers in the network config, they will not work!")

        self.layers = nn.ModuleDict()
        in_channels = 3  # Initial number of channels (e.g., RGB)
        current_size = in_size
        film_idx = 0
        for i, config in enumerate(conv_config):
            layer_name = f"conv_{i}"
            padding = config.get('padding', 0)
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels, config['filters'], kernel_size=config['kernel_size'], stride=config['stride'], padding=padding),
                #nn.BatchNorm2d(config['filters']),
                nn.Mish()
            )
            self.layers[layer_name] = conv_layer
            in_channels = config['filters']  # Update in_channels for the next layer

            # Update current_size for each layer
            current_size = [(size - config['kernel_size'] + 2 * padding) // config['stride'] + 1 for size in current_size]

            self.final_channels = in_channels
            self.final_size = current_size
            if config.get('film', False):
                film_layer_name = f"film_{film_idx}"
                self.layers[film_layer_name] = FiLM()  # Add FiLM layer if specified in config
                film_idx += 1
        
        if not use_ssm:
            # Calculate the number of input features for the linear layer
            linear_input_features = in_channels * current_size[0] * current_size[1]
            #self.output = nn.Linear(linear_input_features, 2*latent_size) 
            current_size = linear_input_features
        else:
            self.ssm = SpatialSoftmax(current_size[0], current_size[1], temperature=ssm_temp)
            #self.output = nn.Linear(in_channels*2, 2*latent_size)
            current_size = in_channels*2

        # Calculate the splits required for the beta and gamma parameters provided by the FiLM generator network
        self.beta_gamma_splits = [config['filters'] for config in conv_config if config.get('film', False)]

        if fc_sizes is not None:
            self.fc_layers = nn.ModuleDict()
            for i, fc_size in enumerate(fc_sizes):
                layer_name=f"dense_{i}"
                fc_layer = nn.Sequential(
                    nn.Linear(current_size, fc_size),
                    nn.Mish()
                )
                self.fc_layers[layer_name] = fc_layer
                self.fc_layers[f"fc_film_{film_idx}"] = FiLM()
                film_idx += 1
                current_size = fc_size
            self.beta_gamma_splits.extend(fc_sizes)
        else:
            self.fc_layers = None
        self.output = nn.Linear(current_size, 2*latent_size)

    def get_final_size(self):
        if self.use_ssm:
            return (self.ssm.width, self.ssm.height)
        else:
            return self.final_size

    def get_ssm_size(self):
        if self.use_ssm:
            return (self.ssm.width, self.ssm.height)
        else:
            return None

    def get_final_channels(self):
        if self.use_ssm:
            return self.output.in_features
        else:
            return self.final_channels
        
    def get_ssm_channels(self):
        if self.use_ssm:
            return self.output.in_features
        else:
            return None

    def forward(self, x, v):
        betas, gammas = self.film_gen(v)  # Get FiLM params
        betas = torch.split(betas, self.beta_gamma_splits, dim=1)
        gammas = torch.split(gammas, self.beta_gamma_splits, dim=1)
        
        bg_idx = 0  # Index for betas and gammas
        for layer_name, layer in self.layers.items():
            if isinstance(layer, FiLM):
                x = layer(x, gammas[bg_idx], betas[bg_idx])
                bg_idx += 1
            else:
                x = layer(x)
        
        if self.use_ssm:
            keys, attn_map = self.ssm(x)
            x = keys.flatten(start_dim=1)
        else:
            x = x.flatten(start_dim=1)

        if self.fc_layers is not None:
            for layer_name, layer in self.fc_layers.items():
                if isinstance(layer, FiLM):
                    x = layer(x, gammas[bg_idx], betas[bg_idx])
                    bg_idx += 1
                else:
                    x = layer(x)
        
        x = self.output(x)

        return x

class LikelihoodModel(nn.Module):
    def __init__(self, latent_size, aux_size, conv_config, in_channels=8, in_size=(4, 4), target_img_size=(32,32), filmgen_hidden_dim=[64], use_inverse_ssm=False, out_channels=3, fc_sizes = None):
        super(LikelihoodModel, self).__init__()
        self.in_channels = in_channels
        self.use_inverse_ssm = use_inverse_ssm
        # Initialize the FiLM generator
        if fc_sizes is not None:
            fc_total_size = sum(fc_sizes)
        else:
            fc_total_size = 0
        filmgen_output_size = sum([config['filters'] * 2 for config in conv_config if config.get('film', False)]) + (fc_total_size*2)
        self.film_gen = FiLMGen(aux_size, filmgen_output_size, filmgen_hidden_dim)
        # Calculate the splits required for the beta and gamma parameters provided by the FiLM generator network

        self.beta_gamma_splits = []

        film_idx = 0
        current_size = latent_size
        if fc_sizes is not None:
            self.fc_layers = nn.ModuleDict()
            self.beta_gamma_splits = fc_sizes
            for i, fc_size in enumerate(fc_sizes):
                layer_name=f"dense_{i}"
                fc_layer = nn.Sequential(
                    nn.Linear(current_size, fc_size),
                    nn.Mish()
                )
                self.fc_layers[layer_name] = fc_layer
                self.fc_layers[f"fc_film_{film_idx}"] = FiLM()
                film_idx += 1
                current_size = fc_size
        else:
            self.fc_layers = None
        if use_inverse_ssm:
            self.in_size = (1,1)
            self.inverse_ssm = InverseSpatialSoftmax(in_size[0], in_size[1])
            in_channels = self.in_channels // 2
        else:
            self.in_size = in_size

        # Initial linear layer to create feature map
        self.input = nn.Sequential(
            nn.Linear(current_size, self.in_size[0] * self.in_size[1] * self.in_channels),  # Adjust channel size as needed
            nn.Mish(),
        )

        self.layers = nn.ModuleDict()
        current_size = in_size

        for i, config in enumerate(conv_config):
            # Adjust layer configurations to use transposed convolutions
            layer_type = config.get('type', 'deconv')
            layer_name = f"{layer_type}_{i}"
            stride = config.get('stride', 1)
            padding = config.get('padding', 0)
            output_padding = config.get('output_padding', 0)
            kernel_size = config['kernel_size']
            if layer_type == 'deconv':
                layer = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, config['filters'], kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
                    #nn.BatchNorm2d(config['filters']),
                    nn.Mish()
                )
            elif layer_type == 'upscale_conv':
                sf = config.get('scale_factor', 2)
                layer = nn.Sequential(
                    nn.Upsample(scale_factor=sf),
                    nn.Conv2d(in_channels, config['filters'], kernel_size=kernel_size, stride=stride, padding=padding),
                    #nn.BatchNorm2d(config['filters']),
                    nn.Mish()
                )
            elif layer_type == 'conv':
                layer = nn.Sequential(
                    nn.Conv2d(in_channels, config['filters'], kernel_size=kernel_size, stride=stride, padding=padding),
                    #nn.BatchNorm2d(config['filters']),
                    nn.Mish()
                )
            elif layer_type == 'deconv_conv':
                layer = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, config['filters'], kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
                    nn.Mish(),
                    nn.Conv2d(in_channels, config['filters'], kernel_size=1, stride=1, padding=0),
                    #nn.BatchNorm2d(config['filters']),
                    nn.Mish()
                )

            self.layers[layer_name] = layer
            in_channels = config['filters']

            # Update current_size for each layer
            current_size = [stride * (size - 1) + kernel_size - 2 * padding for size in current_size]

            # Add FiLM layers if specified
            if config.get('film', False):
                film_layer_name = f"film_{film_idx}"
                film_idx += 1
                self.layers[film_layer_name] = FiLM()
            # add batchnorm
            
        #self.smoothing = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.beta_gamma_splits.extend([config['filters'] for config in conv_config if config.get('film', False)])
        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),  # Reduce to 3 channels
            nn.Sigmoid()  # Apply sigmoid activation
        )

    def forward(self, x: Tensor, v):
        batch_dim = x.size(0)
        betas, gammas = self.film_gen(torch.concat([v, x], dim=1))  # Get FiLM params
        betas = torch.split(betas, self.beta_gamma_splits, dim=1)
        gammas = torch.split(gammas, self.beta_gamma_splits, dim=1)
        bg_idx = 0
        if self.fc_layers is not None:
            for layer_name, layer in self.fc_layers.items():
                if isinstance(layer, FiLM):
                    # print(f"{layer_name}, {x.shape}, {gammas[bg_idx].shape}")
                    x = layer(x, gammas[bg_idx], betas[bg_idx])
                    bg_idx += 1
                else:
                    x = layer(x)
        x = self.input(x)
        if self.use_inverse_ssm:
            keys = x.view(batch_dim, self.in_channels // 2, 2)
            x = self.inverse_ssm(keys)
        else:
            x = x.view(batch_dim, 
                    self.in_channels, 
                    self.in_size[0],
                    self.in_size[1])  # Adjust dimensions as needed

        for layer_name, layer in self.layers.items():
            if isinstance(layer, FiLM):
                x = layer(x, gammas[bg_idx], betas[bg_idx])
                bg_idx += 1
            else:
                x = layer(x)
                #print(f"{layer_name} {x.shape}")
        #x = self.smoothing(x)
        x = self.output(x)
        return x

class VAE(nn.Module):
    def __init__(self, 
                 latent_size, 
                 aux_size, 
                 encoder_config, 
                 decoder_config, 
                 img_size=(32, 32), 
                 decoder_in_channels=8, 
                 decoder_in_size=(4, 4), 
                 encoder_filmgen_hidden_dim=[64], 
                 decoder_filmgen_hidden_dim=[64], 
                 state_update_rule='bayesian', 
                 device='cpu', 
                 use_ssm=False, 
                 use_inverse_ssm=False, 
                 ssm_temperature=0.5, 
                 skip_initial_prior=False,
                 encoder_fc_sizes=None, 
                 decoder_fc_sizes=None):
        
        super(VAE, self).__init__()

        self.latent_size = latent_size 
        self.state_update_rule = state_update_rule
        self.device = device

        # Encoder and Decoder
        self.encoder = PosteriorModel(latent_size=self.latent_size, aux_size=aux_size, conv_config=encoder_config,
                                      in_size=img_size, filmgen_hidden_dim=encoder_filmgen_hidden_dim, 
                                      use_ssm=use_ssm, ssm_temp=ssm_temperature, fc_sizes=encoder_fc_sizes).to(device)

        if not (use_ssm and not use_inverse_ssm):
            decoder_in_size = self.encoder.get_final_size()
            decoder_in_channels = self.encoder.get_final_channels()

        self.decoder = LikelihoodModel(latent_size=self.latent_size, aux_size=self.latent_size + aux_size, conv_config=decoder_config,
                                       in_channels=decoder_in_channels, in_size=decoder_in_size, 
                                       target_img_size=img_size, filmgen_hidden_dim=decoder_filmgen_hidden_dim, 
                                       use_inverse_ssm=use_inverse_ssm, fc_sizes=decoder_fc_sizes).to(device)

        # Initial prior
        self.initial_latent_distribution = D.Normal(torch.zeros(self.latent_size, device=device),
                                                    torch.ones(self.latent_size, device=device))
        self.latent_distribution = self.initial_latent_distribution

        self.skip_initial_prior = skip_initial_prior
        self.latent_initialised = False
        
        # LSTM for recurrent state handling (if chosen)
        if self.state_update_rule == 'lstm':
            self.lstm = nn.LSTM(input_size=2*self.latent_size, hidden_size=2*self.latent_size, batch_first=True).to(device)
            self.lstm_hidden = None

        print(self)

    def reset_state(self):
        self.latent_distribution = self.initial_latent_distribution
        self.latent_initialised = False
        if self.state_update_rule == 'lstm':
            self.lstm_hidden = None  # Reset LSTM state

    def belief_update(self, x):
        if (not self.latent_initialised) and self.skip_initial_prior:
            self.latent_initialised = True
            return x.split(self.latent_size, dim=1)
        # Recurrent state handling
        if self.state_update_rule == 'bayesian':
            mu, logvar = x.split(self.latent_size, dim=1)
            logvar = torch.clamp(logvar, max=8)
            var = torch.exp(logvar)
            mu, var = self.bayes_update(mu, var, self.latent_distribution.mean, self.latent_distribution.variance)
        elif self.state_update_rule == 'lstm':
            mu, var = self.lstm_update(x)
        elif self.state_update_rule == 'additive':
            mu, logvar = x.split(self.latent_size, dim=1)
            logvar = torch.clamp(logvar, max=8)
            var = torch.exp(logvar)
            mu, var = self.additive_update(mu, var, self.latent_distribution.mean, self.latent_distribution.variance)
        elif self.state_update_rule == 'ema':
            # do ema
            return NotImplementedError
        else: # No belief updates, don't use recurrent state
            mu, logvar = x.split(self.latent_size, dim=1)
            logvar = torch.clamp(logvar, max=8)
            var = torch.exp(logvar)
        return mu, var

    def bayes_update(self, mu_1, var_1, mu_2, var_2):
        # Update latent state using Gaussian filtering
        mu = ((var_2 * mu_1) + (var_1 * mu_2)) / (var_1 + var_2)
        var = 1 / ((1/var_1) + (1/var_2))
        return [mu, var]
    
    def lstm_update(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 

        if self.lstm_hidden is None:
            batch_size = x.size(0)
            h0 = torch.zeros(1, batch_size, 2*self.latent_size).to(self.device)
            c0 = torch.zeros(1, batch_size, 2*self.latent_size).to(self.device)
            self.lstm_hidden = (h0, c0)

        lstm_out, self.lstm_hidden = self.lstm(x, self.lstm_hidden)
        lstm_out = lstm_out.squeeze(1)

        mu, logvar = torch.split(lstm_out, self.latent_size, dim=1)
        var = torch.exp(logvar)

        return mu, var

    def additive_update(self, mu_1, var_1, mu_2, var_2):
        return [mu_1+mu_2, var_1+var_2]

    def forward(self, x, v):
        latent_distribution, z = self.encode(x, v)
        x_decoded = self.decode(z, v)
        return latent_distribution, x_decoded, z

    def encode(self, x, v, imagine=False):
        x_encoded = self.encoder(x, v)
        
        mu, var = self.belief_update(x_encoded)

        var = torch.clip(var, 0.1)
        latent_distribution = D.Normal(mu, torch.sqrt(var))
        z = latent_distribution.rsample()

        if not imagine and self.state_update_rule != 'no_aggregation':
            self.latent_distribution = D.Normal(mu.detach(), torch.sqrt(var).detach())

        return latent_distribution, z

    def decode(self, z, v):
        return self.decoder(z, v)

class CatVAE(nn.Module):
    def __init__(self, 
                 cat_size, 
                 num_cats,
                 aux_size, 
                 encoder_config, 
                 decoder_config, 
                 img_size=(32, 32), 
                 decoder_in_channels=8, 
                 decoder_in_size=(4, 4), 
                 encoder_filmgen_hidden_dim=[64], 
                 decoder_filmgen_hidden_dim=[64], 
                 state_update_rule='lstm', 
                 device='cpu', 
                 use_ssm=False, 
                 use_inverse_ssm=False, 
                 ssm_temperature=0.5, 
                 skip_initial_prior=False,
                 encoder_fc_sizes=None, 
                 decoder_fc_sizes=None):
        
        super(CatVAE, self).__init__()
        self.cat_size = cat_size
        self.num_cats = num_cats
        latent_size = num_cats
        half_latent_size = int(num_cats // 2)
        self.state_update_rule = state_update_rule
        self.device = device

        # Encoder and Decoder
        self.encoder = PosteriorModel(latent_size=half_latent_size * cat_size, aux_size=aux_size, conv_config=encoder_config,
                                      in_size=img_size, filmgen_hidden_dim=encoder_filmgen_hidden_dim, 
                                      use_ssm=use_ssm, ssm_temp=ssm_temperature, fc_sizes=encoder_fc_sizes).to(device)
        
        decoder_in_size = self.encoder.get_final_size()
        decoder_in_channels = self.encoder.get_final_channels()
        self.decoder = LikelihoodModel(latent_size=num_cats*cat_size, aux_size=(num_cats*cat_size) + aux_size, conv_config=decoder_config,
                                       in_channels=decoder_in_channels, in_size=decoder_in_size, 
                                       target_img_size=img_size, filmgen_hidden_dim=decoder_filmgen_hidden_dim, 
                                       use_inverse_ssm=use_inverse_ssm, fc_sizes=decoder_fc_sizes).to(device)

        # Initial prior
        self.initial_latent_distribution = D.Categorical(torch.ones([self.cat_size, self.num_cats]).to(device))
        self.latent_distribution = self.initial_latent_distribution

        self.skip_initial_prior = skip_initial_prior
        self.latent_initialised = False
        
        # LSTM for recurrent state handling (if chosen)
        if self.state_update_rule == 'lstm':
            self.lstm = nn.LSTM(input_size=self.num_cats * self.cat_size, hidden_size=self.num_cats * self.cat_size, batch_first=True).to(device)
            self.lstm_hidden = None

        print(self)

    def reset_state(self):
        self.latent_distribution = self.initial_latent_distribution
        self.latent_initialised = False
        if self.state_update_rule == 'lstm':
            self.lstm_hidden = None  # Reset LSTM state

    def belief_update(self, x):
        if (not self.latent_initialised) and self.skip_initial_prior:
            self.latent_initialised = True
            return x
        # Recurrent state handling
        if self.state_update_rule == 'bayesian':
            return NotImplementedError
        elif self.state_update_rule == 'lstm':
            new_probs = self.lstm_update(x)
        elif self.state_update_rule == 'additive':
            new_probs = self.additive_update(x, self.latent_distribution.mean, self.latent_distribution.variance)
        elif self.state_update_rule == 'ema':
            # do ema
            return NotImplementedError
        else: # No belief updates, don't use recurrent state
            new_probs = x
        return new_probs

    def bayes_update(self, mu_1, var_1, mu_2, var_2):
        # Update latent state using Gaussian filtering
        mu = ((var_2 * mu_1) + (var_1 * mu_2)) / (var_1 + var_2)
        var = 1 / ((1/var_1) + (1/var_2))
        return [mu, var]
    
    def lstm_update(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 

        if self.lstm_hidden is None:
            batch_size = x.size(0)
            h0 = torch.zeros(1, batch_size, self.num_cats * self.cat_size).to(self.device)
            c0 = torch.zeros(1, batch_size, self.num_cats * self.cat_size).to(self.device)
            self.lstm_hidden = (h0, c0)

        lstm_out, self.lstm_hidden = self.lstm(x, self.lstm_hidden)
        lstm_out = lstm_out.squeeze(1)

        return lstm_out

    def additive_update(self, mu_1, var_1, mu_2, var_2):
        return [mu_1+mu_2, var_1+var_2]

    def forward(self, x, v):
        latent_distribution, z = self.encode(x, v)
        x_decoded = self.decode(z, v)
        return latent_distribution, x_decoded, z

    def encode(self, x, v, imagine=False):
        x_encoded = self.encoder(x, v)
        new_probs = self.belief_update(x_encoded)
        new_probs = torch.softmax(new_probs.reshape(new_probs.size(0), self.cat_size, self.num_cats),dim=2)
        latent_distribution = D.Categorical(new_probs)
        z = gumbel_softmax(latent_distribution.logits, self.cat_size, self.num_cats, 0.05, False)
        if not imagine and self.state_update_rule != 'no_aggregation':
            self.latent_distribution = D.Categorical(new_probs)

        return latent_distribution, z

    def decode(self, z, v):
        return self.decoder(z, v)
