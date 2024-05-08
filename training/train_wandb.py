from active_vision_model import VAE, SceneDataset, FreeEnergyLoss, SOM, VAEController
from torch import Tensor
import loss_schedulers, model_hooks
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import random
import time
import math
import wandb
import sys, traceback


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_RETRIES = 5

NUM_EPOCHS = 600

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

EXAMPLE_INDEX = 0
LOGGING_FREQ = 100
EVAL_FREQ = 500

EVAL_SCENE_INDICES = [0, 10, 20]
EVAL_FINAL_INDICES = [31, 21, 15]
EVAL_CONTEXT_INDICES = [0, 6, 12, 18, 24, 30, 36, 42, 53, 34]

MAX_LOGS_PER_EPOCH = 1
MAX_EVALS_PER_EPOCH = 1
NUM_EVALS = 3

DEFAULT_ENCODER = [{'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False}]
DEFAULT_DECODER = [{'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False}]

def init_weights(m):
    if isinstance(m, nn.Linear):
        # For Linear layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # For Convolutional layers
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def reshape_to_heatmap(tensor):
    length = tensor.numel()
    sqrt_len = int(math.sqrt(length))

    # Find the factors of length closest to its square root
    for i in range(sqrt_len, 0, -1):
        if length % i == 0:
            factor1 = i
            factor2 = length // i
            break

    # Reshape the tensor
    return tensor.view(1, factor1, factor2)

def enhance_image(images):
    # Define the transformations
    tensor_to_pil = transforms.ToPILImage()  # Handles single images only
    pil_to_tensor = transforms.PILToTensor()
    
    enhanced_images = []

    # Process each image in the batch
    for i in range(images.size(0)):  # Iterate over the batch dimension
        image = images[i]
        pil_image = tensor_to_pil(image)
        smoothed_image = pil_image.filter(ImageFilter.SMOOTH)
        enhancer = ImageEnhance.Contrast(smoothed_image)
        contrast_enhanced_image = enhancer.enhance(1.5)
        edge_enhanced_image = contrast_enhanced_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        tensor_image = pil_to_tensor(edge_enhanced_image)
        enhanced_images.append(tensor_image)

    # Stack the list of tensors along a new dimension, effectively creating a batch
    enhanced_images_batch = torch.stack(enhanced_images) / 255.0

    return enhanced_images_batch

def is_multi_nested(lst):
    # Check if the input is a list
    if not isinstance(lst, list):
        return False
    # Iterate through the list to check for nested lists
    for item in lst:
        if isinstance(item, list):
            return True  # Found a nested list
    return False  # No nested lists found

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

def log_image_to_wandb(image:Tensor, image_name:str, image_caption=None):
    image = image.cpu().detach().squeeze().permute(1, 2, 0)  # Reshape for plotting (32, 32, 3)
    image_np = image.numpy()  # Convert to numpy array
    wandb.log({f"{image_name}" : wandb.Image(image_np, caption=image_caption)}, commit=False)

def get_controller(model, temp, som:SOM=None):
    X = np.linspace(X_RANGE[0], X_RANGE[1], CONTROLLER_NUM_X)
    Y = np.linspace(Y_RANGE[0], Y_RANGE[1], CONTROLLER_NUM_Y)
    loss_fn = FreeEnergyLoss(mse_reduction_mode='sum').to(DEVICE)
    return VAEController(model.to(DEVICE), X, Y, None, efe_fn=loss_fn.to(DEVICE), device=DEVICE, temperature=temp, som=som)

def get_test_grid(controller, eval_obs, eval_views, som:SOM=None): # TODO: FIX THE ORDERING OF THE IMAGE GRID
    with torch.no_grad():
        o = eval_obs.squeeze(1)
        o = enhance_image(o)
        v = eval_views.squeeze(1)
        print(o.shape)
        if som is not None:
            v = som.get_activations(v)
        efes, probs, o_hat = controller.plan(o.to(DEVICE), v.to(DEVICE))
        reversed_tensor = torch.flip(o_hat, [1])

        for idx in range(efes.size(0)):
            # Create a grid of images
            images = reversed_tensor[idx]
            c, r, ch, h, w = images.shape
            images = images.transpose(0, 1).reshape(c * r, ch, h, w)
            grid = vutils.make_grid(images, nrow=CONTROLLER_NUM_X)
            efe_heatmap = efes[idx].transpose(0,1).cpu().detach().numpy()
            #log_image_to_wandb(o[idx], f"eval/input_{idx}", f"input_image_{idx}")
            wandb.log({f"eval/reconstructed_viewpoints_{idx}": wandb.Image(grid, caption=f"reconstructed_viewpoints_{idx}")}, commit=False)
            wandb.log({f"eval/efe_heatmap_{idx}": wandb.Image(efe_heatmap, caption=f"efe_heatmap_{idx}")}, commit=False)

def create_hook_function(hook, name_prefix, name):
    return lambda module, input, output: hook.hook(module, input, output, f"{name_prefix}/{name}") 

def register_model_hooks(model:VAE, use_ssm, use_inverse_ssm):
    # Initialise model hooks
    activation_hook = model_hooks.ActivationHook(example_index=EXAMPLE_INDEX)
    feature_map_hook = model_hooks.FeatureMapHook(example_index=EXAMPLE_INDEX)
    ssm_hook = model_hooks.SpatialSoftmaxHook(example_index=EXAMPLE_INDEX)
    
    # Register hooks
    #  Encoder hooks
    for name, layer in model.encoder.layers.items():
        print(f"Registering hook for {name}")
        hook_function = create_hook_function(feature_map_hook, "encoder", name)
        layer.register_forward_hook(hook_function)
    if use_ssm:
        model.encoder.ssm.register_forward_hook(lambda module, input, output: ssm_hook.hook(module, input, output, "encoder/ssm"))
        # TODO Add possibility of FiLM layers after SSM
    model.encoder.register_forward_hook(lambda module, input, output: activation_hook.hook(module, input, output, "encoder/output"))

    #  Decoder hooks
    if use_inverse_ssm:
        model.decoder.inverse_ssm.register_forward_hook(lambda module, input, output: feature_map_hook.hook(module, input, output, "decoder/inverse_ssm"))
    for name, layer in model.decoder.layers.items():
        print(f"Registering hook for {name}")
        hook_function = create_hook_function(feature_map_hook, "decoder", name)
        layer.register_forward_hook(hook_function)

    return activation_hook, feature_map_hook, ssm_hook

def eval_view_reconstruction(model, context_obs, context_views, query_obs, query_views, som=None):
    with torch.no_grad():
        # first show the model the context
        for frame in range(context_views.size(1)): 
            o = context_obs[:, frame, :, :, :]
            o = enhance_image(o)
            v = context_views[:,frame,:]
            if som is not None:
                v = som.get_activations(v)
            latent_D, o_hat, z = model(o.to(DEVICE), v.to(DEVICE))
        # then generate new viewpoints after inputting the final observation
        controller = get_controller(model, CONTROLLER_TEMP, som=som)
        get_test_grid(controller, query_obs, query_views, som=som)

def run_experiment(**config):

    mdl_name = wandb.run.name
    print(mdl_name)
    # Initialise config parameters
    lr=config.get("learning_rate", 1e-4)
    batch_size=config.get("batch_size", 16)
    img_size = config.get("img_size", (32,32))
    img_scale = config.get("img_scale", 1)
    dataset_name = config.get("dataset_name", "scenes_data_simplebg2")
    manual_seed = config.get("manual_seed", None)
    min_subset_size = config.get("min_subset_size", 3) 
    max_subset_size = config.get("max_subset_size", 10)
    var=config.get("output_variance", 0.65)
    rec_config = config.get('rec_config', {'max_weight':1.})
    reg_config = config.get('reg_config', {'max_weight':1.})
    base_pl = config.get('base_pl', 0)
    base_sl = config.get('base_sl', 0)
    pl_config = config.get('pl_config', {'max_weight':1.})
    sl_config = config.get('sl_config', {'max_weight':1.})
    num_prompts = config.get('num_prompts', 1)
    use_ssm = config.get('use_ssm', False)
    use_inverse_ssm = config.get('use_inverse_ssm', False)
    num_epochs = config.get('num_epochs', NUM_EPOCHS)
    img_stdev = config.get('img_stdev', 0.1)
    use_img_prepro = config.get('use_img_prepro', False)
    # Initialise loss schedulers
    rec_scheduler = get_scheduler_from_config(**rec_config)
    reg_scheduler = get_scheduler_from_config(**reg_config)
    pl_scheduler = get_scheduler_from_config(**pl_config)
    sl_scheduler = get_scheduler_from_config(**sl_config)

    # Initialise model, optimiser and loss function
    model = get_model(**config)
    model_name = mdl_name
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = FreeEnergyLoss(mse_reduction_mode='sum').to(DEVICE)

    # Register hooks
    activation_hook, feature_map_hook, ssm_hook = register_model_hooks(model, use_ssm, use_inverse_ssm)

    gen = torch.Generator()
    # Initialise dataset
    if manual_seed is not None:
        # Use fixed seed on train test split for reproducability 
        gen.manual_seed(manual_seed)

    train_dataset = SceneDataset(dataset_dir=f'data/actvis/{dataset_name}/scenes', img_size=img_size, scale_factor=img_scale, load_into_memory=True)
    test_dataset = SceneDataset(dataset_dir=f'data/actvis/{dataset_name}_test/scenes', img_size=img_size, scale_factor=img_scale, load_into_memory=True)
    print(DEVICE)
    kwargs = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 4} if DEVICE == 'cuda' else {}
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    print(train_dataloader.prefetch_factor)
    # Get eval sets
    # Extract scenes
    selected_scenes = [test_dataset[i] for i in EVAL_SCENE_INDICES]

    # Extract context observations and views
    eval_context_obs = torch.stack([scene[0][EVAL_CONTEXT_INDICES] for scene in selected_scenes]).to(DEVICE)
    eval_context_views = torch.stack([scene[1][EVAL_CONTEXT_INDICES] for scene in selected_scenes]).to(DEVICE)
    for idx, context_obs in enumerate(eval_context_obs):
            grid = vutils.make_grid(context_obs)
            wandb.log({f"eval/context_{idx}": wandb.Image(grid, caption=f"context_{idx}")}, commit=False)

    # Extract final query observations and views
    eval_final_obs = torch.stack([scene[0][index].unsqueeze(0) for scene, index in zip(selected_scenes, EVAL_FINAL_INDICES)]).to(DEVICE)
    eval_final_view = torch.stack([scene[1][index].unsqueeze(0) for scene, index in zip(selected_scenes, EVAL_FINAL_INDICES)]).to(DEVICE)

    # Initialise SOM, if needed
    use_som = config.get('som', False)
    if use_som:
        som_config = config['som_config']
        som = get_som(**som_config)
    else:
        som = None

    best_loss = 9999999

    for epoch in range(num_epochs):
        
        et0 = time.time()

        train_step = 0
        train_shaped_rec_sum, train_shaped_reg_sum, train_shaped_pl_sum, train_shaped_sl_sum, train_shaped_loss_sum = 0,0,0,0,0
        train_rec_sum, train_reg_sum, train_pl_sum, train_sl_sum, train_loss_sum = 0,0,0,0,0

        test_step = 0
        test_shaped_rec_sum, test_shaped_reg_sum, test_shaped_pl_sum, test_shaped_sl_sum, test_shaped_loss_sum = 0,0,0,0,0
        test_rec_sum, test_reg_sum, test_pl_sum, test_sl_sum, test_loss_sum = 0,0,0,0,0

        log_counter = 0
        eval_counter = 0

        # Training loop
        for idx, batch in enumerate(train_dataloader):
            optimiser.zero_grad()
            scene_obs, scene_views = batch
            scene_len = scene_views.shape[1]
            indices = torch.randperm(scene_len)
            subset_size = random.randint(min_subset_size, max_subset_size)
            subset_indices = indices[:subset_size]
            prompt_start_index = subset_size
            prompt_end_index = prompt_start_index + num_prompts
            prompt_indices = indices[prompt_start_index:prompt_end_index]
            subset_obs = scene_obs[:,subset_indices]
            subset_views = scene_views[:,subset_indices]
            prompt_obs = scene_obs[:,prompt_indices]
            prompt_view = scene_views[:,prompt_indices]
            rec_loss = 0.0
            d_list = []
            activation_hook.should_log = False
            feature_map_hook.should_log = False
            if use_ssm:
                ssm_hook.should_log = False

            for frame in range(subset_views.size(1)): # first pass: encode only
                o = subset_obs[:,frame, :, :, :]
                if use_img_prepro:
                    o = enhance_image(o)
                noise = torch.randn_like(o) * img_stdev 
                noisy_o = o + noise
                torch.clamp(noisy_o, 0, 1)
                v = subset_views[:,frame,:]
                if use_som:
                    v = som.get_activations(v)
                latent_D, _ = model.encode(noisy_o.to(DEVICE), v.to(DEVICE))
                d_list.append(latent_D) 
            reg_loss = loss_fn.regularization_term(latent_D)
            s = latent_D.rsample()
            for frame in range(subset_views.size(1)): # second pass: decode only
                o = subset_obs[:,frame, :, :, :]
                if use_img_prepro:
                    o = enhance_image(o)
                v = subset_views[:,frame,:]
                if use_som:
                    v = som.get_activations(v)
                o_hat = model.decode(s, v.to(DEVICE))
                rec = loss_fn.reconstruction_term(o.to(DEVICE), o_hat.to(DEVICE), var=var)
                rec_loss += rec
                d_list.append(latent_D) 

            #Prompt a new viewpoint
            if base_pl != 0:
                p_loss_tot = torch.tensor(0.0).to(DEVICE)
                for prompt_index in range(num_prompts):
                    prompt_v = prompt_view[:,prompt_index]
                    prompt_o = prompt_obs[:,prompt_index]
                    if use_som:
                        prompt_v = som.get_activations(prompt_v)
                    z = latent_D.rsample()
                    o_hat = model.decode(z, prompt_v.to(DEVICE))
                    _, p_loss = loss_fn(prompt_o.to(DEVICE), o_hat.to(DEVICE), latent_D, var=var)
                    p_loss_tot += p_loss
                p_loss = base_pl * p_loss_tot / num_prompts
            else:
                p_loss = torch.tensor(0).to(DEVICE)

            if base_sl != 0:
                smoothing_losses = []
                for i in range(len(d_list) - 1, 0, -1):
                    curr_D = d_list[i]
                    prev_D = d_list[i-1]
                    new_mu, new_var = model.bayes_update(prev_D.mean, prev_D.variance, curr_D.mean, curr_D.variance)
                    smoothed_D = D.Normal(new_mu, torch.sqrt(new_var))
                    smoothing_losses.append(D.kl_divergence(prev_D, smoothed_D).sum())
                s_loss = sum(smoothing_losses) / (subset_size-1)
                s_loss *= base_sl
            else:
                s_loss = torch.tensor(0.0).to(DEVICE)
            #loss, geco_lam = gec.shape_loss(cumulated_rec, reg)
            shaped_p_loss = pl_scheduler.get_weight(epoch) * p_loss
            shaped_s_loss = sl_scheduler.get_weight(epoch) * s_loss
            shaped_rec_loss = rec_scheduler.get_weight(epoch) * rec_loss
            shaped_reg_loss = reg_scheduler.get_weight(epoch) * reg_loss
            loss = shaped_s_loss + shaped_p_loss + shaped_rec_loss + shaped_reg_loss
            loss.backward()
            optimiser.step()
            unshaped_loss = p_loss + s_loss + rec_loss + reg_loss
            model.reset_state()

            train_shaped_loss_sum += loss.item()
            train_shaped_pl_sum += shaped_p_loss.item()
            train_shaped_sl_sum += shaped_s_loss.item()
            train_shaped_rec_sum += shaped_rec_loss.item()
            train_shaped_reg_sum += shaped_reg_loss.item()

            train_loss_sum += unshaped_loss.item()
            train_pl_sum += p_loss.item()
            train_sl_sum += s_loss.item()
            train_rec_sum += rec_loss.item()
            train_reg_sum += reg_loss.item()

            train_step += 1

        # Testing loop
        for idx, scene in enumerate(test_dataloader):
            with torch.no_grad():
                
                scene_obs, scene_views = scene
                scene_len = scene_views.shape[1]
                indices = torch.randperm(scene_len)
                subset_size = random.randint(min_subset_size, max_subset_size)
                subset_indices = indices[:subset_size]
                prompt_index = indices[subset_size]
                prompt_start_index = subset_size
                prompt_end_index = prompt_start_index + num_prompts
                prompt_indices = indices[prompt_start_index:prompt_end_index]
                subset_obs = scene_obs[:,subset_indices]
                subset_views = scene_views[:,subset_indices]
                prompt_obs = scene_obs[:,prompt_indices]
                prompt_view = scene_views[:,prompt_indices]
                if (epoch+1) % EVAL_FREQ == 0 and eval_counter < MAX_EVALS_PER_EPOCH:
                    eval_this_step = True
                    eval_counter +=1
                else:
                    eval_this_step = False

                rec_loss = 0.0
                d_list = []
                time_step = 0

                if (epoch+1) % LOGGING_FREQ == 0 and log_counter < MAX_LOGS_PER_EPOCH:
                    log_this_step = True
                    log_counter += 1
                else:
                    log_this_step = False

                for frame in range(subset_views.size(1)): # First pass: get scene representaion
                    if log_this_step and frame==(subset_views.size(1)-1):
                        log_this_frame = True
                        activation_hook.should_log = True
                        feature_map_hook.should_log = True
                        if use_ssm:
                            ssm_hook.should_log = True
                    else:
                        log_this_frame = False

                    o = subset_obs[:, frame, :, :, :]
                    if use_img_prepro:
                        o = enhance_image(o)
                    noise = torch.randn_like(o) * img_stdev
                    noisy_o = o + noise
                    torch.clamp(noisy_o, 0, 1)
                    v = subset_views[:,frame,:]
                    if use_som:
                        v = som.get_activations(v)
                    latent_D, z = model.encode(noisy_o.to(DEVICE), v.to(DEVICE))
                    #print(latent_D.mean.shape)
                    
                    d_list.append(latent_D)
                    time_step += 1
                    #activations = {**activation_hook.data, **feature_map_hook.data}
                    activation_hook.should_log = False
                    feature_map_hook.should_log = False
                    if use_ssm:
                        #activations = {**activations, **ssm_hook.data}
                        ssm_hook.should_log = False

                reg_loss = loss_fn.regularization_term(latent_D)
                s = latent_D.rsample()

                for frame in range(subset_views.size(1)): # Second pass: reconstruction
                    if log_this_step and frame==(subset_views.size(1)-1):
                        log_this_frame = True
                        activation_hook.should_log = True
                        feature_map_hook.should_log = True
                        if use_ssm:
                            ssm_hook.should_log = True
                    else:
                        log_this_frame = False

                    o = subset_obs[:, frame, :, :, :]
                    if use_img_prepro:
                        o = enhance_image(o)
                    mean = 0
                    std = 0.1
                    noise = torch.randn_like(o) * std + mean
                    noisy_o = o + noise
                    torch.clamp(noisy_o, 0, 1)
                    if log_this_frame:
                        log_image_to_wandb(o[EXAMPLE_INDEX], "model_state/input", image_caption="input")
                    v = subset_views[:,frame,:]
                    if use_som:
                        v = som.get_activations(v)
                    o_hat = model.decode(s, v.to(DEVICE))
                    if log_this_frame:
                        latent_means = latent_D.mean[EXAMPLE_INDEX].detach().cpu()
                        latent_vars = latent_D.variance[EXAMPLE_INDEX].detach().cpu()
                        latent_vector = s[EXAMPLE_INDEX].detach().cpu()
                        latent_state_data = {f"model_state/latent_distribution_mean" : wandb.Histogram(latent_means.clone().numpy()),
                                   f"model_state/latent_distribution_var": wandb.Histogram(latent_vars.clone().numpy()),
                                   f"model_state/latent_vector": wandb.Histogram(latent_vector.clone().numpy())}
                        means_heatmap = reshape_to_heatmap(latent_means)
                        vars_heatmap = reshape_to_heatmap(latent_vars)
                        z_heatmap = reshape_to_heatmap(latent_vector)
                        wandb.log(latent_state_data, commit=False)
                        wandb.log({"model_state/latent_mu_heatmap": wandb.Image(means_heatmap, caption="Latent Means"),
                                   "model_state/latent_var_heatmap": wandb.Image(vars_heatmap, caption="Latent Variances"),
                                   "model_state/latent_sample_heatmap": wandb.Image(z_heatmap, caption="Latent Sample")}, commit=False)
                        log_image_to_wandb(o_hat[EXAMPLE_INDEX], "model_state/reconstruction", image_caption="reconstruction")
                    #print(latent_D.mean.shape)
                    rec = loss_fn.reconstruction_term(o.to(DEVICE), o_hat.to(DEVICE), var=var)
                    rec_loss += rec #Accumulate only reconstruction error
                    time_step += 1

                    #activations = {**activation_hook.data, **feature_map_hook.data}
                    activation_hook.should_log = False
                    feature_map_hook.should_log = False
                    if use_ssm:
                        #activations = {**activations, **ssm_hook.data}
                        ssm_hook.should_log = False


                if base_pl != 0:
                    p_loss_tot = torch.tensor(0.0).to(DEVICE)
                    for prompt_index in range(num_prompts):
                        prompt_v = prompt_view[:,prompt_index]
                        prompt_o = prompt_obs[:,prompt_index]
                        if use_som:
                            prompt_v = som.get_activations(prompt_v)
                        z = latent_D.rsample()
                        o_hat = model.decode(z, prompt_v.to(DEVICE))
                        _, p_loss = loss_fn(prompt_o.to(DEVICE), o_hat.to(DEVICE), latent_D, var=var)
                        p_loss_tot += p_loss
                        if log_this_step:
                            prompt_viewpoint_data = {"prompt/viewpoint" : prompt_view[EXAMPLE_INDEX].detach().cpu().numpy()[EXAMPLE_INDEX]}
                            wandb.log(prompt_viewpoint_data, commit=False)
                            log_image_to_wandb(o_hat[EXAMPLE_INDEX], "prompt/reconstruction", image_caption="prompt_reconstruction")
                    p_loss = base_pl * p_loss_tot / num_prompts
                else:
                    p_loss = torch.tensor(0.0).to(DEVICE)

                if eval_this_step:
                    model.reset_state()
                    eval_view_reconstruction(model, eval_context_obs, eval_context_views, eval_final_obs, eval_final_view, som=som)

                if base_sl != 0:
                    smoothing_losses = []
                    for i in range(len(d_list) - 1, 0, -1):
                        curr_D = d_list[i]
                        prev_D = d_list[i-1]
                        new_mu, new_var = model.bayes_update(prev_D.mean, prev_D.variance, curr_D.mean, curr_D.variance)
                        smoothed_D = D.Normal(new_mu, torch.sqrt(new_var))
                        smoothing_losses.append(D.kl_divergence(prev_D, smoothed_D).sum())
                    s_loss = sum(smoothing_losses) / (subset_size-1)
                    s_loss *= base_sl
                else:
                    s_loss = torch.tensor(0).to(DEVICE)

                shaped_p_loss = pl_scheduler.get_weight(epoch) * p_loss
                shaped_s_loss = sl_scheduler.get_weight(epoch) * s_loss
                shaped_rec_loss = rec_scheduler.get_weight(epoch) * rec_loss
                shaped_reg_loss = reg_scheduler.get_weight(epoch) * reg_loss
                loss = shaped_s_loss + shaped_p_loss + shaped_rec_loss + shaped_reg_loss
                
                unshaped_loss = p_loss + s_loss + rec_loss + reg_loss
                model.reset_state()

                test_shaped_loss_sum += loss.item()
                test_shaped_pl_sum += shaped_p_loss.item()
                test_shaped_sl_sum += shaped_s_loss.item()
                test_shaped_rec_sum += shaped_rec_loss.item()
                test_shaped_reg_sum += shaped_reg_loss.item()

                test_loss_sum += unshaped_loss.item()
                test_pl_sum += p_loss.item()
                test_sl_sum += s_loss.item()
                test_rec_sum += rec_loss.item()
                test_reg_sum += reg_loss.item()

                test_step += 1

        # Calculate average losses
        train_shaped_rec_avg = train_shaped_rec_sum / train_step
        train_shaped_reg_avg = train_shaped_reg_sum / train_step
        train_shaped_pl_avg = train_shaped_pl_sum / train_step
        train_shaped_sl_avg = train_shaped_sl_sum / train_step
        train_shaped_loss_avg = train_shaped_loss_sum / train_step

        train_rec_avg = train_rec_sum / train_step
        train_reg_avg = train_reg_sum / train_step
        train_pl_avg = train_pl_sum / train_step
        train_sl_avg = train_sl_sum / train_step
        train_loss_avg = train_loss_sum / train_step

        test_shaped_rec_avg = test_shaped_rec_sum / test_step
        test_shaped_reg_avg = test_shaped_reg_sum / test_step
        test_shaped_pl_avg = test_shaped_pl_sum / test_step
        test_shaped_sl_avg = test_shaped_sl_sum / test_step
        test_shaped_loss_avg = test_shaped_loss_sum / test_step

        test_rec_avg = test_rec_sum / test_step
        test_reg_avg = test_reg_sum / test_step
        test_pl_avg = test_pl_sum / test_step
        test_sl_avg = test_sl_sum / test_step
        test_loss_avg = test_loss_sum / test_step

        train_metrics = {
            "train/loss":train_shaped_loss_avg, 
            "train/rec_loss":train_shaped_rec_avg, 
            "train/reg_loss":train_shaped_reg_avg, 
            "train/unshaped/loss":train_loss_avg, 
            "train/unshaped/rec_loss":train_rec_avg, 
            "train/unshaped/reg_loss":train_reg_avg, 
        }
        test_metrics = {
            "test/loss":test_shaped_loss_avg, 
            "test/rec_loss":test_shaped_rec_avg, 
            "test/reg_loss":test_shaped_reg_avg, 
            "test/unshaped/loss":test_loss_avg,
            "test/unshaped/rec_loss":test_rec_avg, 
            "test/unshaped/reg_loss":test_reg_avg,
        }
        if base_pl != 0:
            train_metrics["train/p_loss"] = train_shaped_pl_avg
            train_metrics["train/unshaped/p_loss"] = train_pl_avg
            test_metrics["test/p_loss"] = test_shaped_pl_avg
            test_metrics["test/unshaped/p_loss"] = test_pl_avg
        if base_sl != 0:
            train_metrics["train/s_loss"] = train_shaped_sl_avg
            train_metrics["train/unshaped/s_loss"] = train_sl_avg
            test_metrics["test/s_loss"] = test_shaped_sl_avg
            test_metrics["test/unshaped/s_loss"] = test_sl_avg

        metrics = {"epoch":epoch, **train_metrics, **test_metrics}

        wandb.log(metrics)
        print(f"EPOCH {epoch} | TRAIN {train_metrics} | TEST {test_metrics}")

        # Update best model and save
        if test_loss_avg < best_loss:
            best_loss = test_loss_avg
            torch.save(model.state_dict(), f"models/new/{model_name}_best.pt")

        torch.save(model.state_dict(), f"models/new/{model_name}_latest.pt")
        et1 = time.time()
        print(f'Epoch time: {et1-et0}')

def init_experiment(wandb_config, manual_seed=None):
    # Initialize a new wandb run
    wandb.init(
        # set the wandb project where this run will be logged
        project="active-vision",
        # track hyperparameters and run metadata
        config=wandb_config
    )

    run_experiment(**wandb_config)

    # Finish the wandb run
    wandb.finish()

def wandb_sweep_run():
    # Initialize a new wandb run
    wandb.init()
    
    try:
        run_experiment(**wandb.config)
    except ValueError as e:
        wandb.log({"error": str(e)})
        wandb.log({"traceback": traceback.format_exc()})
    except TypeError as e:
        print(traceback.print_exc())
        wandb.log({"error": str(e)})
        wandb.log({"traceback": traceback.format_exc()})
    except RuntimeError as e:
        print(traceback.format_exc())
    except Exception as e:
        print(traceback.format_exc())
    finally:
        wandb.finish()

    # Finish the wandb run
    wandb.finish()

def perform_sweep(config, count=None):
    sweep_id = wandb.sweep(config, project="active-vision")
    wandb.agent(sweep_id, wandb_sweep_run, count=count)

from config import network_configs, sweep_configs
if __name__ == '__main__':
    config = network_configs.massive_sweep
    sweep_id = wandb.sweep(config, project="active-vision")
    wandb.agent(sweep_id, wandb_sweep_run, count=120)
    config = network_configs.massive_sweep64
    sweep_id = wandb.sweep(config, project="active-vision")
    wandb.agent(sweep_id, wandb_sweep_run, count=120)