from src.data_retrieval.faiss_search import MyFaiss
from src.data_encoder.clip_module.model import CLIP_Model

import warnings
warnings.filterwarnings("ignore")


clip_model =  CLIP_Model(device = "cuda")

search = MyFaiss(bin_path = "/home/toonies/Learn/Text_Video_Retrieval/data/dicts/faiss_CLIP_cosine.bin",
                    json_path = "/home/toonies/Learn/Text_Video_Retrieval/data/dicts/keyframes_id_search.json",
                    encoder_model= clip_model)

text_query = "Hai nguoi dang dung noi chuyen"
search.search_query(text_query, k=12)
