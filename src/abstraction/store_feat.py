from abc import ABC, abstractmethod


class StoreFeat(ABC):
    @abstractmethod
    def __init__(self, model, *args):
        pass    
    
    @abstractmethod
    def save_image2npy(self, text_file, npy_path) -> None:
        pass
    
    @abstractmethod
    def save_text2npy(self, text_file, npy_path) -> None:
        pass