from model.model import VAE
from model.loss import FreeEnergyLoss
from model.som import SOM
import loss_schedulers
from constructors import get_model, get_scheduler_from_config, get_som
from dataset.scene_dataset import SceneDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import time
from training.utils import enhance_image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_RETRIES = 5

NUM_EPOCHS = 600

X_RANGE = (-0.65, 0.65) # These are properties of the dataset
Y_RANGE = (0.9, 1.25)

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

EVAL_SCENE_INDICES = [0, 14, 36]
EVAL_FINAL_INDICES = [31, 21, 15]
EVAL_CONTEXT_INDICES = [0, 6, 12, 18, 24, 30, 36, 42, 53, 34]

MAX_LOGS_PER_EPOCH = 1
MAX_EVALS_PER_EPOCH = 1
NUM_EVALS = 3

DEFAULT_ENCODER = [{'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False}]
DEFAULT_DECODER = [{'filters': 16, 'kernel_size': 3, 'stride': 2, 'film': False}]

def train(**config):
    mdl_name = config.get("model_name", "actvis")
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
    use_ssm = config.get('use_ssm', False)
    use_inverse_ssm = config.get('use_inverse_ssm', False)
    num_epochs = config.get('num_epochs', NUM_EPOCHS)
    img_stdev = config.get('img_stdev', 0.1)
    use_img_prepro = config.get('use_img_prepro', False)
    # Initialise loss schedulers
    rec_scheduler = get_scheduler_from_config(**rec_config)
    reg_scheduler = get_scheduler_from_config(**reg_config)

    # Initialise model, optimiser and loss function
    model = get_model(**config)
    model_name = mdl_name
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = FreeEnergyLoss(mse_reduction_mode='sum').to(DEVICE)

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

        # Training loop
        for idx, batch in enumerate(train_dataloader):
            optimiser.zero_grad()
            scene_obs, scene_views = batch
            scene_len = scene_views.shape[1]
            indices = torch.randperm(scene_len)
            subset_size = random.randint(min_subset_size, max_subset_size)
            subset_indices = indices[:subset_size]
            subset_obs = scene_obs[:,subset_indices]
            subset_views = scene_views[:,subset_indices]
            rec_loss = 0.0
            d_list = []

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

            shaped_rec_loss = rec_scheduler.get_weight(epoch) * rec_loss
            shaped_reg_loss = reg_scheduler.get_weight(epoch) * reg_loss
            loss = shaped_rec_loss + shaped_reg_loss
            loss.backward()
            optimiser.step()
            unshaped_loss = rec_loss + reg_loss
            model.reset_state()

            train_shaped_loss_sum += loss.item()
            train_shaped_rec_sum += shaped_rec_loss.item()
            train_shaped_reg_sum += shaped_reg_loss.item()

            train_loss_sum += unshaped_loss.item()
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
                subset_obs = scene_obs[:,subset_indices]
                subset_views = scene_views[:,subset_indices]

                rec_loss = 0.0
                d_list = []
                time_step = 0

                for frame in range(subset_views.size(1)): # First pass: get scene representaion
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

                reg_loss = loss_fn.regularization_term(latent_D)
                s = latent_D.rsample()

                for frame in range(subset_views.size(1)): # Second pass: reconstruction
                    o = subset_obs[:, frame, :, :, :]
                    if use_img_prepro:
                        o = enhance_image(o)
                    mean = 0
                    std = 0.1
                    noise = torch.randn_like(o) * std + mean
                    noisy_o = o + noise
                    torch.clamp(noisy_o, 0, 1)
                    v = subset_views[:,frame,:]
                    if use_som:
                        v = som.get_activations(v)
                    o_hat = model.decode(s, v.to(DEVICE))

                    #print(latent_D.mean.shape)
                    rec = loss_fn.reconstruction_term(o.to(DEVICE), o_hat.to(DEVICE), var=var)
                    rec_loss += rec #Accumulate only reconstruction error
                    time_step += 1

                shaped_rec_loss = rec_scheduler.get_weight(epoch) * rec_loss
                shaped_reg_loss = reg_scheduler.get_weight(epoch) * reg_loss
                loss = shaped_rec_loss + shaped_reg_loss
                
                unshaped_loss = rec_loss + reg_loss
                model.reset_state()

                test_shaped_loss_sum += loss.item()
                test_shaped_rec_sum += shaped_rec_loss.item()
                test_shaped_reg_sum += shaped_reg_loss.item()

                test_loss_sum += unshaped_loss.item()
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
        print(f"EPOCH {epoch} | TRAIN {train_metrics} | TEST {test_metrics}")

        # Update best model and save
        if test_loss_avg < best_loss:
            best_loss = test_loss_avg
            torch.save(model.state_dict(), f"models/new/{model_name}_best.pt")

        torch.save(model.state_dict(), f"models/new/{model_name}_latest.pt")
        et1 = time.time()
        print(f'Epoch time: {et1-et0}')