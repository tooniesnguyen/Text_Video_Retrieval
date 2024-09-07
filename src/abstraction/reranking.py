from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

class RerankMethod (ABC):
    def __init__(self, device: str, **kwargs):
        pass
    
    @abstractmethod
    def reranking_result(self, images_file: List[str], text_query: str, scores: List[int]):
        pass