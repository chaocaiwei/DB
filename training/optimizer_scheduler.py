import torch

from concern.config import Configurable, State


class OptimizerScheduler(Configurable):
    optimizer = State()
    optimizer_args = State(default={})
    learning_rate = State(autoload=False)

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        optimizer_args = kwargs['optimizer_args']
        learning_rate = kwargs['learning_rate']
        learning_rate['lr'] = optimizer_args['lr']
        if 'lr' in cmd:
            self.optimizer_args['lr'] = cmd['lr']
        if 'epochs' in cmd:
            learning_rate['epochs'] = cmd['epochs']
        self.load('learning_rate', cmd=cmd, **kwargs)

    def create_optimizer(self, parameters):
        optimizer = getattr(torch.optim, self.optimizer)(
                parameters, **self.optimizer_args)
        if hasattr(self.learning_rate, 'prepare'):
            self.learning_rate.prepare(optimizer)
        return optimizer
