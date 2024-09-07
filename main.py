from src.service.data_retrieval.faiss_search import SearchFaiss
# from src.service.data_encoder.InternVideo2 import EncoderInternVideo2
from src.service.data_encoder.BLIP2 import BLIP2Model
import torch
import os


import warnings
warnings.filterwarnings("ignore")



if __name__ == "__main__":
    clip_model =  BLIP2Model(device = "cuda")

    search = SearchFaiss(bin_path = "/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/data/dicts/bin/faiss_BLIP2_cosine.bin",
                        json_path = "/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/data/dicts/json/keyframes_id_search.json",
                        encoder_model= clip_model)

    text_query = "Một ngừơi mặc áo trắng đang dẫn chương trình với một người phụ nữ"
    img_paths = search.search_query(text_query, k =42, rerank=True)

    search.save_result(img_paths, save_path = "./results")