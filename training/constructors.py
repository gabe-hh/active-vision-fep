from model.model import VAE
from model.loss import FreeEnergyLoss
from model.som import SOM
from controller.model_controller import VAEController
import loss_schedulers
import numpy as np
import torch
from training.utils import init_weights

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

X_RANGE = (-0.65, 0.65) # These are properties of the dataset
Y_RANGE = (0.9, 1.25)
CONTROLLER_NUM_X = 8
CONTROLLER_NUM_Y = 4
CONTROLLER_TEMP = 1.

SCHEDULER_MAPPING = {
    None: loss_schedulers.StraightThroughScheduler,
    'StepScheduler': loss_schedulers.StepScheduler,
    'LinearGrowthScheduler': loss_schedulers.LinearGrowthScheduler,
    'InverseExponentialGrowthScheduler': loss_schedulers.InverseExponentialGrowthScheduler,
    'SigmoidGrowthScheduler': loss_schedulers.SigmoidGrowthScheduler,
    'CosineGrowthScheduler': loss_schedulers.CosineGrowthScheduler,
    'LinearDecayScheduler': loss_schedulers.LinearDecayScheduler,
    'ExponentialDecayScheduler': loss_schedulers.ExponentialDecayScheduler,
    'InverseTimeDecayScheduler': loss_schedulers.InverseTimeDecayScheduler,
    'CosineAnnealingScheduler': loss_schedulers.CosineAnnealingScheduler
}

DEFAULT_ENCODER = [{'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False}]
DEFAULT_DECODER = [{'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False}]

def get_model(**config):
    latent_size = config.get("latent_size", 128)
    aux_size = config.get("aux_size", 2)
    encoder_config = config.get("encoder_config", DEFAULT_ENCODER)
    encoder_filmgen_hidden_dim = config.get("encoder_filmgen_hidden_dim", [64,64])
    encoder_fc_sizes = config.get("encoder_fc_sizes", None)
    decoder_config = config.get("decoder_config", DEFAULT_DECODER)
    print(encoder_config)
    decoder_in_channels = config.get("decoder_in_channels", 3)
    decoder_in_size = config.get("decoder_in_size", (4,4))
    decoder_fc_sizes = config.get("decoder_fc_sizes", None)
    if isinstance(decoder_in_size, list): # If wandb has converted to list
        # Convert to tuple
        decoder_in_size = tuple(decoder_in_size)
    decoder_filmgen_hidden_dim = config.get("decoder_filmgen_hidden_dim", [64,64])
    img_size = config.get("img_size", (32,32))
    img_scale = config.get("img_scale", 1)
    scaled_img_size = (int(img_size[0] // (1/img_scale)), int(img_size[1] // (1/img_scale)))
    state_update_rule = config.get("state_update_rule", "bayesian")
    if config.get('som', False):
        assert aux_size == config.get('som_size', 32), "Auxilliary data size must match SOM size if using SOM!"

    use_ssm = config.get("use_ssm", False)
    use_inverse_ssm = config.get("use_inverse_ssm", False)
    ssm_temp = config.get("ssm_temp", 0.5)

    model = VAE(latent_size, aux_size, encoder_config, decoder_config, img_size=scaled_img_size, 
                decoder_in_channels=decoder_in_channels, decoder_in_size=decoder_in_size, 
                encoder_filmgen_hidden_dim=encoder_filmgen_hidden_dim, decoder_filmgen_hidden_dim=decoder_filmgen_hidden_dim,
                state_update_rule=state_update_rule, device=DEVICE, 
                use_ssm=use_ssm, use_inverse_ssm=use_inverse_ssm, ssm_temperature=ssm_temp, skip_initial_prior=True,
                encoder_fc_sizes=encoder_fc_sizes, decoder_fc_sizes=decoder_fc_sizes)
    
    model.apply(init_weights)
    return model.to(DEVICE)

def get_som(**config):
    size = config.get('size', 32)
    num_y = config.get('num_y', 4)
    num_x = config.get('num_x', size//num_y)
    gamma = config.get('gamma', 140)
    X = np.linspace(X_RANGE[0], X_RANGE[1], num_x)
    Y = np.linspace(Y_RANGE[0], Y_RANGE[1], num_y)
    ref_vectors = np.stack(np.meshgrid(X, Y, indexing='ij'), -1).reshape(-1, 2)
    ref_vectors = torch.from_numpy(ref_vectors).to(torch.float32)
    som = SOM(ref_vectors=ref_vectors, gamma=gamma)
    return som

def get_scheduler_from_config(**config) -> loss_schedulers.BaseScheduler:
    scheduler_name = config.get('scheduler', None)
    scheduler_class = SCHEDULER_MAPPING.get(scheduler_name)
    scheduler_args = {k: v for k, v in config.items() if k != 'scheduler'}
    return scheduler_class(**scheduler_args)

def get_controller(model, temp, som:SOM=None):
    X = np.linspace(X_RANGE[0], X_RANGE[1], CONTROLLER_NUM_X)
    Y = np.linspace(Y_RANGE[0], Y_RANGE[1], CONTROLLER_NUM_Y)
    loss_fn = FreeEnergyLoss(mse_reduction_mode='sum').to(DEVICE)
    return VAEController(model.to(DEVICE), X, Y, None, efe_fn=loss_fn.to(DEVICE), device=DEVICE, temperature=temp, som=som)
