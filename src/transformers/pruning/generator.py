import torch
from torch import nn
from torch.nn import Parameter, init
from petl.petl_factory import Adapter_Layer, LoRALayer, Linear

def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf

def weights(module):
    buffers = [buff for _, buff in module.named_buffers()]
    for name, param in module.named_parameters():
        for buff in buffers:
            if param.size() == buff.size():
                yield param

def trainable(module):
    r"""Returns boolean whether a module is trainable.
    """
    return not isinstance(module, (Identity1d, Identity2d))

def prunable(named_module):
    r"""Returns boolean whether a module is prunable.
    """
    name, module = named_module[0], named_module[1]
    return isinstance(module, (Adapter_Layer, LoRALayer, Linear))

def parameters(model):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    for name, module in filter(lambda p: trainable(p), model.modules()):
        for param in module.parameters(recurse=False):
            yield param

def masked_parameters(model):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for name, module in filter(lambda p: prunable(p), model.module.named_modules()):
        for mask, param in zip(masks(module), weights(module)):
            yield mask, param


class Identity1d(nn.Module):
    def __init__(self, num_features):
        super(Identity1d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W


class Identity2d(nn.Module):
    def __init__(self, num_features):
        super(Identity2d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features, 1, 1))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W
