from abc import ABC, abstractmethod
from typing import Dict, List
from src.abstraction.encoder_model import EncoderModel

class SearchDB(ABC):
    
    @abstractmethod
    def __init__(self, bin_path: str, json_path: str, 
                 model: EncoderModel, show_time_compute: bool):
        pass
    
    
    @abstractmethod
    def search_query(self, query: str, k: int) -> List:
        
        pass
        
    @abstractmethod
    def show_images(self, image_paths) -> None:
        pass
    

