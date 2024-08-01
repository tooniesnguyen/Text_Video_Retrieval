import yaml
from typing import Dict, Any
import numpy as np

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data

def read_txt_to_array(txt_file: str) -> np.ndarray:
    """
    Visualize the result of inference
    
    Parameters:
    -----------
    - `txt_file` (str): path of txt file

    Returns:
    --------
    - The contents of the txt file as an array
    
    """
    with open(txt_file, 'r') as f:
        arr = [list(map(int, line.strip('[]\n ').split())) for line in f if line.strip()]
    return np.array(arr)

