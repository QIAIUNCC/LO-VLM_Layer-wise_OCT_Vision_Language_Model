"""
utils.py
--------
Helper functions for training.
"""


def count_trainable_params(model):
    """
    Count number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
