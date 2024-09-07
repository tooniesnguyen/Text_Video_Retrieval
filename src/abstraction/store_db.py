from abc import ABC, abstractmethod

class StoreDB(ABC):
    def __init__(self, model_name, feat_shape: int, method: str, **kwargs):
        pass

    @abstractmethod
    def merge_npy2bin(self, npy_path: str, bin_path: str):
        pass
    
