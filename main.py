from src.service.data_retrieval.faiss_search import SearchFaiss
from src.service.data_encoder.CLIP import CLIPModel
import torch
import os


import warnings
warnings.filterwarnings("ignore")



if __name__ == "__main__":
    clip_model =  CLIPModel(device = "cuda")

    search = SearchFaiss(bin_path = "/home/toonies/Learn/Text_Video_Retrieval/data/dicts/bin/faiss_CLIP_cosine.bin",
                        json_path = "/home/toonies/Learn/Text_Video_Retrieval/data/dicts/json/keyframes_id_search.json",
                        encoder_model= clip_model)

    text_query = "The man is wearing a red shirt and yellow hat"
    img_paths = search.search_query(text_query, k =42, rerank=False)

    search.save_result(img_paths, save_path = "./results")