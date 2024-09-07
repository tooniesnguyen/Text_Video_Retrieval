from src.abstraction.reranking import RerankMethod
import src.service.reranking.ImageReward.ImageReward as RM
from typing import List
import torch

class ImageRewardMethod(RerankMethod):
    def __init__(self, device, **kwargs):
        self.device = device
        self.model = self.load_model()
        
    def load_model(self):
        return RM.load("ImageReward-v1.0").to(self.device)
    
    def reranking_result(self, images_path: str, text_query: str, scores: None) -> List:
        with torch.no_grad():
            ranking, rewards = self.model.inference_rank(text_query, images_path)
            # print("\nPreference predictions:\n")
            # print(f"ranking = {ranking}")
            # print(f"rewards = {rewards}")
            sorted_arr_path_img = [images_path[i - 1] for i in ranking]
        
        return sorted_arr_path_img
        