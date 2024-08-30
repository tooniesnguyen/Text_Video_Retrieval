from src.service.data_retrieval.faiss_search import SearchFaiss
from src.service.data_encoder.InternVideo2 import EncoderInternVideo2
import torch
import os


import warnings
warnings.filterwarnings("ignore")



if __name__ == "__main__":
    clip_model =  EncoderInternVideo2(device = "cuda")

    search = SearchFaiss(bin_path = "/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/data/dicts/bin/faiss_InternVideo2_cosine.bin",
                        json_path = "/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/data/dicts/json/keyframes_id_search.json",
                        encoder_model= clip_model)

    text_query = "The man is wearing a red shirt and yellow hat"
    img_paths = search.search_query(text_query, k =42, rerank=False)

    search.save_result(img_paths, save_path = "./results")