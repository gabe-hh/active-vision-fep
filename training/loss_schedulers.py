import math

class BaseScheduler:
    def __init__(self):
        pass

    def get_weight(self, current_epoch):
        raise NotImplementedError("This method should be implemented by subclasses.")

class StraightThroughScheduler(BaseScheduler):
    def __init__(self, max_weight=1., min_weight=None, growth_rate=None, num_epochs=None):
        self.max_weight = max_weight

    def get_weight(self, current_epoch):
        return self.max_weight

class StepScheduler(BaseScheduler):
    def __init__(self, min_weight=0.1, step_size=10, growth_factor=2):
        self.min_weight = min_weight
        self.step_size = step_size
        self.growth_factor = growth_factor

    def get_weight(self, current_epoch):
        return self.min_weight * (self.growth_factor ** (current_epoch // self.step_size))

# Growth Schedulers
class LinearGrowthScheduler(BaseScheduler):
    def __init__(self, min_weight=0.1, max_weight=1.0, num_epochs=500, growth_rate=None):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.num_epochs = num_epochs

    def get_weight(self, current_epoch):
        return self.min_weight + (current_epoch / self.num_epochs) * (self.max_weight - self.min_weight)

class InverseExponentialGrowthScheduler(BaseScheduler):
    def __init__(self, min_weight=0.1, growth_rate=0.05, max_weight=None, num_epochs=None):
        self.min_weight = min_weight
        self.growth_rate = growth_rate

    def get_weight(self, current_epoch):
        return self.min_weight + (1 - math.exp(-self.growth_rate * current_epoch))

class SigmoidGrowthScheduler(BaseScheduler):
    def __init__(self, max_weight=1.0, growth_rate=0.1, num_epochs=500, min_weight=None):
        self.max_weight = max_weight
        self.growth_rate = growth_rate
        self.num_epochs = num_epochs

    def get_weight(self, current_epoch):
        return self.max_weight / (1 + math.exp(-self.growth_rate * (current_epoch - self.num_epochs / 2)))

class CosineGrowthScheduler(BaseScheduler):
    def __init__(self, min_weight=0.1, max_weight=1.0, num_epochs=500, growth_rate=None):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.num_epochs = num_epochs

    def get_weight(self, current_epoch):
        return self.min_weight + 0.5 * (self.max_weight - self.min_weight) * (1 - math.cos(math.pi * current_epoch / self.num_epochs))

# Decay Schedulers
class LinearDecayScheduler(BaseScheduler):
    def __init__(self, min_weight=0.1, max_weight=1.0, num_epochs=500):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.num_epochs = num_epochs

    def get_weight(self, current_epoch):
        return self.max_weight - (current_epoch / self.num_epochs) * (self.max_weight - self.min_weight)
    
class ExponentialDecayScheduler(BaseScheduler):
    def __init__(self, min_weight=0.1, decay_rate=0.95):
        self.min_weight = min_weight
        self.decay_rate = decay_rate

    def get_weight(self, current_epoch):
        return self.min_weight * (self.decay_rate ** current_epoch)

class InverseTimeDecayScheduler(BaseScheduler):
    def __init__(self, min_weight=1.0, decay_rate=0.1):
        self.min_weight = min_weight
        self.decay_rate = decay_rate

    def get_weight(self, current_epoch):
        return self.min_weight / (1 + self.decay_rate * current_epoch)

class CosineAnnealingScheduler(BaseScheduler):
    def __init__(self, min_weight=0.1, max_weight=1.0, num_epochs=500):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.num_epochs = num_epochs

    def get_weight(self, current_epoch):
        return self.min_weight + 0.5 * (self.max_weight - self.min_weight) * (1 + math.cos(math.pi * current_epoch / self.num_epochs))
