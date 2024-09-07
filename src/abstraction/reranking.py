from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

class RerankMethod(ABC):
    def __init__(self, device: str, **kwargs):
        pass
    
    @abstractmethod
    def rerank_by_id(self, list_ids_scores: Dict[int, float]):
        pass
    
    @abstractmethod
    def rerank_by_files(self, images_file: List[int]):
        pass