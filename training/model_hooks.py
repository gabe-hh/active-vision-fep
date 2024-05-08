import wandb
import torch
import torchvision.utils as vutils
import math

class BaseHook:
    def __init__(self, example_index=0):
        self.example_index = example_index
        self.should_log = False

    def hook(self, module, input, output, layer_name):
        raise NotImplementedError("This method should be implemented by subclasses.")

class ActivationHook(BaseHook):
    def hook(self, module, input, output, layer_name):
        if self.should_log:
            if isinstance(output, torch.Tensor):
                wandb.log({f"{layer_name}_activation" : wandb.Histogram(output.detach().cpu().numpy()[self.example_index])}, commit=False)
            elif isinstance(output, tuple) and all(isinstance(x, torch.Tensor) for x in output):
                wandb.log({f"{layer_name}_activation" : wandb.Histogram(output[0].detach().cpu().numpy()[self.example_index])}, commit=False)

class FeatureMapHook(BaseHook):
    def hook(self, module, input, output, layer_name):
        if self.should_log:
            batch_size, channels, _, _ = output.shape
            #nrow = int(math.ceil(math.sqrt(channels)))
            feature_maps = output.detach().cpu()[self.example_index]
            nrow = math.ceil(math.sqrt(feature_maps.size(0)))
            # Make a grid from the feature map channels
            grid = vutils.make_grid(feature_maps.unsqueeze(1), nrow=nrow, normalize=True, scale_each=False)  # Adjust nrow as needed
            # Log the grid as a single image
            #self.data[f"{layer_name}_feature_maps_grid"] = wandb.Image(grid)
            wandb.log({f"{layer_name}_output":wandb.Image(grid, caption=f"{layer_name} feature maps")}, commit=False)
            # for channel in range(channels):
            #     feature_map = output.detach().cpu()[self.example_index, channel]
            #     self.data[f"{layer_name}_feature_map_{channel}"] = wandb.Image(feature_map)

class SpatialSoftmaxHook(BaseHook):
    def hook(self, module, input, output, layer_name):
        if self.should_log:
            keys, att_map = output
            #self.data[f"{layer_name}_keys_t{self.time_step}"] = wandb.Histogram(keys.detach().cpu().numpy()[self.example_index])
            wandb.log({f"{layer_name}_keys": wandb.Histogram(keys.detach().cpu().numpy()[self.example_index])}, commit=False)
            #batch_size, channels, width, height = att_map.shape
            #nrow = int(math.ceil(math.sqrt(channels)))
            feature_maps = att_map.detach().cpu()[self.example_index]
            nrow = math.ceil(math.sqrt(feature_maps.size(0)))
            # Make a grid from the feature map channels
            grid = vutils.make_grid(feature_maps.unsqueeze(1), nrow=nrow, normalize=True, scale_each=True)  # Adjust nrow as needed

            wandb.log({f"{layer_name}_output": wandb.Image(grid, caption=f"{layer_name} attention maps")}, commit=False)