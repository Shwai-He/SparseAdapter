import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.cuda.amp import autocast
from petl.petl_factory import Adapter_Layer
import numpy as np
import math

class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}
        self.label_smoother = None
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = None


    def score(self, model, dataloader, device, compute_loss=None):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = math.ceil((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)]
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))


    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v ** 2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                kwargs = dict(device=self.device)
                inputs[k] = v.to(**kwargs)

        return inputs


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, dataloader, device, compute_loss=None):
        self.device = device
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)




class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)

    def score(self, model, dataloader, device, compute_loss=None):
        self.device = device
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()
            self.scores[id(p)] = self.scores[id(p)]


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, dataloader, device, compute_loss=None):
        self.device = device
        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            with autocast():
                loss = compute_loss(model, inputs)
                loss = loss.mean()
            loss.backward()
        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        all_scores = torch.where(torch.isinf(all_scores), torch.full_like(all_scores, 1), all_scores)
        maximal = all_scores.max()
        norm = torch.sum(all_scores / maximal)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)
            self.scores[id(p)] = self.scores[id(p)]


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, dataloader, device, compute_loss=None):
        self.device = device
        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            loss = compute_loss(model, inputs, return_outputs=False)
            loss = loss.mean()
            grads = torch.autograd.grad(loss, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            loss = compute_loss(model, inputs, return_outputs=False)
            loss = loss.mean()
            grads = torch.autograd.grad(loss, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()

        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)
            self.scores[id(p)] = self.scores[id(p)]


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, dataloader, device, compute_loss=None):
        self.device = device
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        signs = linearize(model)
        nextiter = next(iter(dataloader))
        input_ids = nextiter['input_ids']
        latent_dim = model.module.base_model.n_embd # roberta model
        outputs = 0
        with autocast():
            for name, module in model.module.named_modules():
                if isinstance(module, Adapter_Layer):
                    inputs = torch.ones([1] + list(input_ids[0, :].shape) + [latent_dim]).to(device)
                    output = module(inputs)
                    outputs += output

            loss = torch.mean(outputs)
        loss.backward()
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            self.scores[id(p)] = self.scores[id(p)]
            p.grad.data.zero_()
        nonlinearize(model, signs)

