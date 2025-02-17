from src.abstraction.reranking import RerankMethod
import src.service.reranking.ImageReward.ImageReward as RM
from src.utils.config import DATA_DIR
from typing import List
import numpy as np
import torch
import os

class ImageRewardMethod(RerankMethod):
    def __init__(self, device, **kwargs):
        self.device = device
        self.model = self.load_model()
        
    def load_model(self):
        return RM.load("ImageReward-v1.0").to(self.device)
    
    def reranking_result(self, images_path: str, text_query: str, scores: None, **kwargs) -> List:
        images_path_root = [os.path.join(DATA_DIR, image_path) for image_path in images_path]
        with torch.no_grad():
            ranking, rewards = self.model.inference_rank(text_query, images_path_root)
            ranking = list(np.array(ranking) - 1)
            print(f"ranking = {ranking}") 
        results_path = np.array(images_path)[ranking].tolist()       
        return results_path
        