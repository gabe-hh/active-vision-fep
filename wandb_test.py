import training.train_wandb
import wandb
import config.network_configs

def run_wandb(config, notes=""):
    # Initialize a new wandb run
    wandb.init(
        # set the wandb project where this run will be logged
        project="active-vision",
        notes=notes,
        # track hyperparameters and run metadata
        config=config
    )
    training.train_wandb.run_experiment(**config)
    # Finish the wandb run
    wandb.finish()

NUM_EPOCHS = 5000

# w/ w/out prepro
# Grow vs decay reg
# 

encoder_config = config.network_configs.enc_3layer_8to32 # No SSM
decoder_config = config.network_configs.dec_3layer_32to8
wandb_config = {
    'reg_config':{
        'max_weight': 1,
    },
    'latent_size': 64,
    'batch_size': 8,
    'img_size': (64,64),
    'img_scale': 0.5,
    'learning_rate': 0.0007,
    "min_subset_size": 3,
    "max_subset_size": 12,
    'output_variance': 0.9,
    'encoder_filmgen_hidden_dim': [64,128],
    'decoder_filmgen_hidden_dim': [66,128],
    'encoder_config': encoder_config,
    'decoder_config': decoder_config,
    'num_epochs': 7000,
    'use_som': False,
    'use_ssm': False,
    'som_gamma': 140,
    'use_inverse_ssm': False,
    'encoder_fc_sizes': [64,64],
    'decoder_fc_sizes': [64,64],
    'img_stdev': 0.12,
    'use_img_prepro': False,
    'manual_seed': 2,
    'dataset_name': 'nobg'
}
run_wandb(wandb_config, notes="No SSM, 3 layer and FC layers")

encoder_config = config.network_configs.enc_3layer_8to32_nofilm # No SSM
decoder_config = config.network_configs.dec_3layer_32to8_nofilm
wandb_config = {
    'reg_config':{
        'max_weight': 1,
    },
    'latent_size': 64,
    'batch_size': 8,
    'img_size': (64,64),
    'img_scale': 0.5,
    'learning_rate': 0.0007,
    "min_subset_size": 3,
    "max_subset_size": 12,
    'output_variance': 0.9,
    'encoder_filmgen_hidden_dim': [64,128],
    'decoder_filmgen_hidden_dim': [66,128],
    'encoder_config': encoder_config,
    'decoder_config': decoder_config,
    'num_epochs': 7000,
    'use_som': False,
    'use_ssm': False,
    'som_gamma': 140,
    'use_inverse_ssm': False,
    'encoder_fc_sizes': [64,64],
    'decoder_fc_sizes': [64,64],
    'img_stdev': 0.12,
    'use_img_prepro': False,
    'manual_seed': 2,
    'dataset_name': 'nobg'
}
run_wandb(wandb_config, notes="No SSM, no FiLM, 3 layer and FC layers")