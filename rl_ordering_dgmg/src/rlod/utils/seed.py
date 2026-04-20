from __future__ import annotations

import os
import random
import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """
    Semente global (python + numpy). Torch entra depois quando começarmos RL/DGMG.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
