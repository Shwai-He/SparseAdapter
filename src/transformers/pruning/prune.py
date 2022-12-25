from tqdm import tqdm
import transformers.pruning.pruners as pruners
import numpy as np

def pruner(method):
    prune_methods = {
        'rand' : pruners.Rand,
        'mag' : pruners.Mag,
        'snip' : pruners.SNIP,
        'grasp': pruners.GraSP,
        'synflow' : pruners.SynFlow,
    }
    return prune_methods[method]

def prune_loop(model, pruner, dataloader, device, sparsity, schedule, scope, epochs, structured=False,
               reinitialize=False, train_mode=False, shuffle=False, invert=False, compute_loss=None):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, dataloader, device, compute_loss)
        if schedule == 'exponential':
            sparse = sparsity ** ((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity) * ((epoch + 1) / epochs)
        # Invert scores
        if invert:
            pruner.invert()
        pruner.mask(sparse, scope)

    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    # if remaining_params - total_params * sparsity >= 5:
    #     print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params * sparsity))
    #     quit()

