from src.abstraction.search_db import SearchDB
from src.abstraction.encoder_model import EncoderModel
from src.utils.timer import time_complexity
from src.utils.translate import Translation

from src.utils.constants import WORK_DIR
# import src.service.reranking.ImageReward.ImageReward as RM

from langdetect import detect
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import math
import torch
import faiss
import json
import os



class SearchFaiss(SearchDB):
    def __init__(self, bin_path: str, json_path: str, 
                 encoder_model = EncoderModel, show_time_compute = True):
        
        self.show_time_compute= show_time_compute
        self.encoder_model = encoder_model
        self.index = self.__load_bin(bin_path)
        self.dict_json = self.__read_json(json_path)
        self.translate = Translation()
        self.rerank = False
        
        
    def __read_json(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
        return data
    
    def __load_bin(self, bin_path):
        return faiss.read_index(bin_path)
    
    def save_result(self, image_paths, save_path: str):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))

        for i in range(1, columns * rows + 1):
            img = plt.imread(image_paths[i - 1])
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))
            plt.imshow(img)
            plt.axis("off")

            # Save each subplot as an individual image file in the "./results" directory
        output_path = os.path.join(save_path, f"result_{self.encoder_model.__class__.__name__}_rerank_{self.rerank}.png")
        plt.savefig(output_path)
        plt.close(fig)
        
    @time_complexity("Reranking Result")
    def __reranking_result(self, images_path: str, prompt: str) -> List:
        model = RM.load("ImageReward-v1.0")
        with torch.no_grad():
            ranking, rewards = model.inference_rank(prompt, images_path)
            print("\nPreference predictions:\n")
            print(f"ranking = {ranking}")
            # print(f"rewards = {rewards}")
            sorted_arr_path_img = [images_path[i - 1] for i in ranking]
        
        return sorted_arr_path_img
    
    
    def search_query(self, text: str, k: int, rerank = False):
        self.rerank = rerank
        if detect(text) == 'vi':
            text = self.translate(text)
        print("Text translation: ", text)
        text_feature = self.encoder_model.text_encoder(text) 
        scores, idx_image = self.index.search(text_feature, k=k)
        print("Idx images", idx_image)
        result_strings = list(map(lambda idx: self.dict_json[idx] if 0 <= idx < len(self.dict_json) else None, idx_image[-1]))
        
        imgs_path_return = [os.path.join(WORK_DIR, "data", image_path) for image_path in result_strings]
        if self.rerank:
            result_strings = self.__reranking_result(result_strings, text)
        
        return result_strings