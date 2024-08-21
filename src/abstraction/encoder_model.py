from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

class EncoderModel(ABC):
    @abstractmethod
    def __init__(self, device):
        pass
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def text_encoder(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def image_encoder(self, image_path: str) -> np.ndarray:
        pass

    