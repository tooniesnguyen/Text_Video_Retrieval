from src.data_retrieval.faiss_search import MyFaiss
from src.data_encoder.CLIP.model import CLIP_Model
import torch
import os


import warnings
warnings.filterwarnings("ignore")



if __name__ == "__main__":
    clip_model =  CLIP_Model(device = "cuda")

    search = MyFaiss(bin_path = "/home/toonies/Learn/Text_Video_Retrieval/data/dicts/bin/faiss_CLIP_cosine.bin",
                        json_path = "/home/toonies/Learn/Text_Video_Retrieval/data/dicts/keyframes_id_search.json",
                        encoder_model= clip_model)

    text_query = "The man is wearing a red shirt and yellow hat"
    img_paths = search.search_query(text_query, k =42, rerank=False)

    search.show_images(img_paths)