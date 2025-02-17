from src.abstraction.search_db import SearchDB
from src.abstraction.encoder_model import EncoderModel
from src.utils.timer import time_complexity
from src.utils.translate import Translation

from src.utils.config import WORK_DIR, DATA_DIR
import src.service.reranking.ImageReward.ImageReward as RM

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
        self.model_not_trans = ["BKAIModel"]
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

        black_image = np.full((256, 256, 3),255)

        for i in range(1, columns * rows + 1):
            try:
                img = plt.imread(os.path.join(DATA_DIR, image_paths[i - 1])) #
            except:
                img = black_image
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))
            plt.imshow(img)
            plt.axis("off") 

            # Save each subplot as an individual image file in the "./results" directory
        output_path = os.path.join(save_path, f"{self.encoder_model.__class__.__name__}_{'rerank' if self.rerank else 'no_rerank'}.png")
        plt.savefig(output_path)
        plt.close(fig)


    @time_complexity("Time search result")
    def search_query(self, text: str, k: int, rerank = None):
        
        self.rerank = rerank
        if detect(text) == 'vi' and self.encoder_model.__class__.__name__ not in self.model_not_trans:
            text = self.translate(text)
        print("Text translation: ", text)
        text_feature = self.encoder_model.text_encoder(text) 
        scores, idx_image = self.index.search(text_feature, k=k)
        scores = scores[0].tolist()
        results_path = list(map(lambda idx: self.dict_json[idx] if 0 <= idx < len(self.dict_json) else None, idx_image[-1]))
        
        # print("Result :", results_path)
        if self.rerank:
            results_path = self.rerank.reranking_result(results_path, text, scores=scores, k=k)
            
        return scores, idx_image, results_path