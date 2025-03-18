import os
import random
import numpy as np
import torch

def set_seed(seed=0):
    """シード値の設定"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def count_unique_elements(data):
    """一意な要素の数を取得"""
    nested_list = data["act_labels"]
    flat_list = [item for sublist in nested_list for item in sublist]
    unique_elements = set(flat_list)
    num_unique_elements = len(unique_elements)

    return num_unique_elements - 1