import network_configs
massive_sweep = {
    "method": "random",  # or "grid", "bayes"
    "metric": {
        "name": "train/unshaped/loss",
        "goal": "minimize"   
    },
    "parameters": {
        "latent_size": {
            "values": [16, 32, 64, 128, 256]
        },
        "learning_rate": {
            "value": 0.0002
        },
        "batch_size": {
            "values": [8, 16, 32]
        },
        "output_variance":{
            "min": 0.3, "max": 1.
        },
        "film_layers":{
            "value": [64, 128]
        },
        "min_subset_size":{
            "min": 2, "max":5
        },
        "max_subset_size":{
            "min": 6, "max": 25
        },
        "num_epochs":{"value": 1800},
        "use_som": {"values": [True, False]},
        "som_gamma": {"value": 140},
        "encoder_config": {"values": network_configs.encoders},
        "decoder_config": {"values": network_configs.decoders}
    },
    "early_terminate": {
        "type": "hyperband",
        "s": 2,
        "eta": 3,
        "max_iter": 27
    }
}

# Define sweep configuration
sweep_config = {
    "method": "random",  # or "grid", "bayes"
    "metric": {
        "name": "test/unshaped/loss",
        "goal": "minimize"   
    },
    "parameters": {
        "latent_size": {
            "min": 96,
            "max": 256
        },
        "learning_rate": {
            "min": 0.00001,
            "max": 0.001
        },
        "batch_size": {
            "values": [8, 16, 32]
        },
        "output_variance":{
            "min": 0.1,
            "max": 1.
        },
        "film_layers":{
            "values": [
                [64],
                [128],
                [64, 64],
                [64, 128],
                [128, 128],
                [64, 32, 64],
                [64, 64, 128]
            ]
        },
        "reg_config": {
            "parameters": {
                "max_weight": {
                    "distribution": "normal",
                    "mu": 1.0,
                    "sigma": 1.0
                }
            }
        },
        "ssm":{
            "values": [True, False]
        },
        "som":{
            "values": [True, False]
        },
        "som_config":{
            "parameters":{
                "size":{"value": 32},
                "num_x":{"value": 8},
                "num_y":{"value": 4},
                "gamma":{
                    "min": 1.,
                    "max": 200.
                }
            }
        }
    }
}

# Define sweep configuration
ssm_test_config = {
    "method": "bayes",  # or "grid", "bayes"
    "metric": {
        "name": "test/unshaped/loss",
        "goal": "minimize"   
    },
    "parameters": {
        "latent_size": {
            "value": 16
        },
        "learning_rate": {
            "value": 0.0003
        },
        "batch_size": {
            "value": 16
        },
        "output_variance":{
            "value": 0.7
        },
        "film_layers":{
            "value": [64, 128]
        },
        "ssm":{
            "value": True
        },
        "inverse_ssm": {
            "values": [True, False]
        },
        "ssm_temp":{
            "distribution": "log_uniform",
            "min": -4,
            "max": 4,
        },
        "som":{
            "value": False
        },
        "min_subset_size":{
            "value": 3
        },
        "max_subset_size":{
            "value": 10
        },
    }
}

ssm_invssm_latent_size_test_config = {
    "method": "grid",  # or "grid", "bayes"
    "metric": {
        "name": "test/unshaped/loss",
        "goal": "minimize"   
    },
    "parameters": {
        "latent_size": {
            "values": [16, 32, 64]
        },
        "learning_rate": {
            "value": 0.0003
        },
        "batch_size": {
            "value": 16
        },
        "output_variance":{
            "value": 0.5
        },
        "film_layers":{
            "value": [64, 128]
        },
        "use_ssm":{
            "value": True
        },
        "use_inverse_ssm": {
            "value": True
        },
        "ssm_temp":{
            "values": [1e-4, 1e-2, 1]
        },
        "min_subset_size":{
            "value": 5
        },
        "max_subset_size":{
            "value": 15
        },
    }
}

test_state_updates_ssm_using_pl_32x32 = {
    "method": "grid",  # or "grid", "bayes"
    "metric": {
        "name": "test/unshaped/p_loss",
        "goal": "minimize"   
    },
    "parameters":{
        "use_som": {
            "value": False
        },
        "use_ssm": {
            "values": [False, True]
        },
        "use_inverse_ssm": {
            "values": [False, True]
        },
        "base_pl": {
            "value": 4
        },
        "base_sl": {
            "value": 0.9478289232844972
        },
        "state_update_rule":{
            "values": ["bayesian", "lstm", "additive"]
        },
        "img_size": {
            "value": (32,32)
        },
        "pl_config": {
            "value": {
            "scheduler": "SigmoidGrowthScheduler",
            "max_weight": 7,
            "min_weight": 0,
            "num_epochs": 600,
            "growth_rate": 0.10571072135660414
            }
        },
        "sl_config": {
            "value": {
            "scheduler": "SigmoidGrowthScheduler",
            "max_weight": 0.7624848683519078,
            "min_weight": 0.1320529614107488,
            "num_epochs": 600,
            "growth_rate": 0.04224560363755421
            }
        },
        "batch_size": {
            "value": 16
        },
        "reg_config": {
            "parameters": {
                "max_weight": {
                    "value": 0.7
                }
            }
        },
        "encoder_filmgen_hidden_dim": {
            "value": [
            128,
            128
            ]
        },
        "decoder_filmgen_hidden_dim": {
            "value": [
            128,
            128
            ]
        },
        "latent_size": {
            "value": 64
        },
        "num_prompts": {
            "value": 6
        },
        "learning_rate": {
            "value": 0.00090503775341834
        },
        "max_subset_size": {
            "value": 22
        },
        "min_subset_size": {
            "value": 4
        },
        "output_variance": {
            "value": 0.7
        }
    }
}
# Define sweep configuration
ssm_test_plsl_config = {
    "method": "random",  # or "grid", "bayes"
    "metric": {
        "name": "test/unshaped/loss",
        "goal": "minimize"   
    },
    "parameters": {
        "latent_size": {
            "value": 128
        },
        "learning_rate": {
            "value": 0.0001
        },
        "batch_size": {
            "value": 16
        },
        "output_variance":{
            "value": 0.5
        },
        "film_layers":{
            "value": [64]
        },
        "reg_config": {
            "parameters": {
                "max_weight": {
                    "value": 1.
                }
            }
        },
        "base_pl":{
            "value": 4.
        },
        "base_sl":{
            "min": 0.,
            "max": 4.
        },
        "pl_config":{
            "parameters": {
                "max_weight": {
                    "min": 3.,
                    "max": 8.},
                "min_weight": {
                    "min": 0.,
                    "max": 3.
                },
                "num_epochs": {"value": 800},
                "growth_rate": {"min": 1e-5, "max": 0.4},
                "scheduler": {
                    "values": [None, "LinearGrowthScheduler", "InverseExponentialGrowthScheduler", "SigmoidGrowthScheduler", "SigmoidGrowthScheduler"]
                }
            }
        },
        "sl_config":{
            "parameters": {
                "max_weight": {
                    "min": 0.5,
                    "max": 6.},
                "min_weight": {
                    "min": 0.,
                    "max": 0.5
                },
                "num_epochs": {"value": 800},
                "growth_rate": {"min": 1e-5, "max": 0.4},
                "scheduler": {
                    "values": [None, "LinearGrowthScheduler", "InverseExponentialGrowthScheduler", "SigmoidGrowthScheduler", "SigmoidGrowthScheduler"]
                }
            }
        },
        "ssm":{
            "value": True
        },
        "inverse_ssm": {
            "values": [True, False]
        },
        "ssm_temp":{
            "min": 1e-6,
            "max": 1000.
        },
        "som":{
            "value": True
        },
        "min_subset_size":{
            "min": 2,
            "max": 6
        },
        "max_subset_size":{
            "min": 6,
            "max": 40
        },
        "som_config":{
            "parameters":{
                "size":{"value": 32},
                "num_x":{"value": 8},
                "num_y":{"value": 4},
                "gamma":{"value": 140}
            }
        }
    }
}

# Define sweep configuration
test_ssm_som_ablation_config = {
    "method": "grid",  # or "grid", "bayes"
    "metric": {
        "name": "test/unshaped/loss",
        "goal": "minimize"   
    },
    "parameters": {
        "latent_size": {
            "value": 128
        },
        "learning_rate": {
            "value": 0.0004
        },
        "batch_size": {
            "value": 16
        },
        "output_variance":{
            "value": 0.7
        },
        "film_layers":{
            "value": [64, 128]
        },
        "ssm":{
            "values": [True, False]
        },
        "inverse_ssm": {
            "values": [True, False]
        },
        "ssm_temp":{
            "value": 0.09
        },
        "som":{
            "values": [True, False]
        },
        "min_subset_size":{
            "value": 3
        },
        "max_subset_size":{
            "value": 12
        },
        "som_config":{
            "parameters":{
                "size":{"value": 32},
                "num_x":{"value": 8},
                "num_y":{"value": 4},
                "gamma":{"value": 140}
            }
        }
    }
}

test_state_updates_32x32 = {
    "method": "grid",  # or "grid", "bayes"
    "metric": {
        "name": "test/unshaped/p_loss",
        "goal": "minimize"   
    },
    "parameters":{
        "use_som": {
            "value": False
        },
        "state_update_rule":{
            "values": ["bayesian", "lstm", "additive"]
        },
        "img_size": {
            "value": (64,64)
        },
        "batch_size": {
            "value": 16
        },
        "reg_config": {
            "parameters": {
                "max_weight": {
                    "value": 0.1
                }
            }
        },
        "encoder_filmgen_hidden_dim": {
            "value": [
            128,
            128
            ]
        },
        "decoder_filmgen_hidden_dim": {
            "value": [
            128,
            128
            ]
        },
        "latent_size": {
            "value": 64
        },
        "learning_rate": {
            "value": 0.00090503775341834
        },
        "max_subset_size": {
            "value": 12
        },
        "min_subset_size": {
            "value": 4
        },
        "output_variance": {
            "value": 0.5
        }
    }
}
# Define sweep configuration
# Define sweep configuration
prompt_test_config = {

    "method": "random",  # or "grid", "bayes"
    "metric": {
        "name": "test/unshaped/p_loss",
        "goal": "minimize"   
    },
    "parameters": {
        "latent_size": {
            "min": 96,
            "max": 256
        },
        "learning_rate": {
            "min": 0.00001,
            "max": 0.001
        },
        "batch_size": {
            "values": [16, 32]
        },
        "output_variance":{
            "value": 0.7
        },
        "film_layers":{
            "values": [
                [16],
                [64],
                [128],
                [32,32],
                [64, 64],
                [64, 128],
                [128, 128],
                [64, 32, 64],
                [64, 64, 128],
                [128, 64, 128]
            ]
        },
        "reg_config": {
            "parameters": {
                "max_weight": {
                    "min": 1e-3,
                    "max": 10.
                }
            }
        },
        "base_pl":{
            "value": 4.
        },
        "num_prompts":{
            "min": 1,
            "max": 5
        },
        "base_sl":{
            "min": 0.,
            "max": 4.
        },
        "pl_config":{
            "parameters": {
                "max_weight": {
                    "min": 3.,
                    "max": 8.},
                "min_weight": {
                    "min": 0.,
                    "max": 3.
                },
                "num_epochs": {"value": 600},
                "growth_rate": {"min": 1e-5, "max": 0.4},
                "scheduler": {
                    "values": [None, "LinearGrowthScheduler", "InverseExponentialGrowthScheduler", "SigmoidGrowthScheduler", "SigmoidGrowthScheduler"]
                }
            }
        },
        "sl_config":{
            "parameters": {
                "max_weight": {
                    "min": 0.5,
                    "max": 6.},
                "min_weight": {
                    "min": 0.,
                    "max": 0.5
                },
                "num_epochs": {"value": 600},
                "growth_rate": {"min": 1e-5, "max": 0.4},
                "scheduler": {
                    "values": [None, "LinearGrowthScheduler", "InverseExponentialGrowthScheduler", "SigmoidGrowthScheduler", "SigmoidGrowthScheduler"]
                }
            }
        },
        "ssm":{
            "values": [True, False]
        },
        "inverse_ssm": {
            "values": [True, False]
        },
        "ssm_temp":{
            "distribution": "log_uniform",
            "min": -6,
            "max": 4.
        },
        "som":{
            "values": [True, False]
        },
        "min_subset_size":{
            "min": 2,
            "max": 10
        },
        "max_subset_size":{
            "min": 10,
            "max": 40
        },
        "som_config":{
            "parameters":{
                "size":{"value": 32},
                "num_x":{"value": 8},
                "num_y":{"value": 4},
                "gamma":{
                    "min": 1.,
                    "max": 200.
                }
            }
        },
        "img_size":{
            "values": [32, 64]
        }
    }
}