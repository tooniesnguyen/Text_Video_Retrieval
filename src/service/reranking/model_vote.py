from typing import List
from src.abstraction.reranking import RerankMethod
import torch.nn as nn

class VotedMethod(RerankMethod):
    def __init__(self, device: str, search_model1, **kwargs):
        self.device = device
        self.search_model1 = search_model1
    

    def most_vote(self, score_model1, result_model1, score_model2, result_model2):
        common_images = list(set(result_model1) & set(result_model2))

        common_images.sort(key=lambda x: score_model1[result_model1.index(x)], reverse=True)

        non_common_model1 = [img for img in result_model1 if img not in common_images]
        non_common_model2 = [img for img in result_model2 if img not in common_images]

        return common_images, non_common_model1, non_common_model2

    def reranking_result(self, 
                         images_file: List[str], 
                         text_query: str, 
                         scores: List[int],
                         k: int):
        scores_model1, _, images_file_model1 = self.search_model1.search_query(text_query, k*2, None)
        common_images, non_common_model1, non_common_model2 = self.most_vote(
                                                                             scores_model1,
                                                                             images_file_model1,
                                                                             scores,
                                                                             images_file)

        len_common_images = len(common_images)
        if len_common_images > k:
            results_path = common_images[:k]
        else:
            len_to_append = k - len_common_images
            results_path = common_images + non_common_model1[:len_to_append]
        return results_path
        