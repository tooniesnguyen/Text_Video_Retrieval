from src.abstraction.search_db import SearchDB
from src.abstraction.encoder_model import EncoderModel
from src.utils.timer import time_complexity
from src.utils.translate import Translation
from langdetect import detect
import matplotlib.pyplot as plt
import numpy as np
import math
import faiss
import json



class MyFaiss(SearchDB):
    def __init__(self, bin_path: str, json_path: str, 
                 encoder_model = EncoderModel, show_time_compute = True):
        
        self.show_time_compute= show_time_compute
        self.encoder_model = encoder_model
        self.index = self.__load_bin(bin_path)
        self.dict_json = self.__read_json(json_path)
        self.translate = Translation()
        
        
    def __read_json(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
        return data
    
    def __load_bin(self, bin_path):
        return faiss.read_index(bin_path)
    
    def show_images(self, image_paths):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))

        for i in range(1, columns*rows +1):
            img = plt.imread(image_paths[i - 1])
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

            plt.imshow(img)
            plt.axis("off")
        plt.show()
    
    def search_query(self, text: str, k: int):
        if detect(text) == 'vi':
            text = self.translate(text)
        print("Text translation: ", text)
        text_feature = self.encoder_model.text_encoder(text) 
        scores, idx_image = self.index.search(text_feature, k=k)
        print("Idx images", idx_image)
        result_strings = list(map(lambda idx: self.dict_json[idx] if 0 <= idx < len(self.dict_json) else None, idx_image[-1]))
        return result_strings