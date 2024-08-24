from src.data_retrieval.faiss_search import MyFaiss
from src.data_encoder.clip_module.model import CLIP_Model
import torch
import os


import warnings
warnings.filterwarnings("ignore")



if __name__ == "__main__":
    clip_model =  CLIP_Model(device = "cuda")

    search = MyFaiss(bin_path = "/home/toonies/Learn/Text_Video_Retrieval/data/dicts/faiss_CLIP_cosine.bin",
                        json_path = "/home/toonies/Learn/Text_Video_Retrieval/data/dicts/keyframes_id_search.json",
                        encoder_model= clip_model)

    text_query = "Một người đàn ông đội nón lá mặc áo đỏ đang ngồi cạnh một người phụ nữ cùng mặc áo đỏ"
    img_paths = search.search_query(text_query, k =48, rerank=True)

    search.show_images(img_paths)
