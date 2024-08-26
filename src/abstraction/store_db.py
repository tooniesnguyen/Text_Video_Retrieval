from abc import ABC, abstractmethod

class StoreDB(ABC):
    @abstractmethod
    def __init__(self, model_name, feat_shape: int, method: str, *args):
        pass

    @abstractmethod
    def merge_npy2bin(self, npy_path: str, bin_path: str):
        pass
    
