from abc import ABC, abstractmethod

class StoreDB(ABC):
    @abstractmethod
    def __init__(self, model_name:str, feat_shape = 512, method = "cosine", *args):
        pass

    @abstractmethod
    def convert_npy2bin(self, npy_path: str, bin_path: str):
        pass