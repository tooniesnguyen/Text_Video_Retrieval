from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

class EncoderModel(ABC):
    def __init__(self, device: str, **kwargs):
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        pass
    
    @abstractmethod
    def text_encoder(self, text: str) -> np.ndarray:
        """
        This function is used to encode the text into features to store 
        or compare in the database

        Parameters
        ----------
        - `text` (str): the text needs to be encode

        Returns
        -------
        - `text_feature` (np.ndarray): the features after the forward encoder block
        """
        
        pass
    
    @abstractmethod
    def image_encoder(self, image_path: str) -> np.ndarray:
        """
        This function is used to encode the image into features to store 
        or compare in the database

        Parameters
        ----------
        - `image_path` (str): the part of the imag. Ex: "../*.jpg"

        Returns
        -------
        - `imge_feature` (np.ndarray): the features after the forward encoder block
        """
        pass


    