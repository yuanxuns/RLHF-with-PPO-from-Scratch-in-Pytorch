import torch
import numpy as np
import random
from transformers import set_seed


def get_device() -> torch.device:
    """
    Get the device to use for training.
    Returns:
        torch.device: The device to use for training.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.has_mps or torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Define a mapping from string to torch dtype
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def seed_everything(seed=2003):
    """
    Seeds all random number generators used in this codebase.

    Args:
        seed: The seed to use for seeding the random number generators.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
