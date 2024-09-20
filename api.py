import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.service.data_retrieval.faiss_search import SearchFaiss
from src.service.data_retrieval.ctrlf_search import CtrlFSearch
from src.service.data_retrieval.ytb_rettrieval import retrieval_video
from src.service.reranking.ImageReward import ImageRewardMethod
from src.service.reranking.model_vote import VotedMethod

from src.service.data_encoder.CLIP import CLIPModel
from src.service.data_encoder.BLIP2 import BLIP2Model
from src.service.data_encoder.InternVideo2 import InternVideo2Model
from src.service.data_encoder.OCR_BKAI import BKAIModel

from src.utils.utils import load_yaml, read_json_file
from src.utils.config import *

import os
import numpy as np
from typing import List, Dict
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title='Ho Chi Minh AI Challenge 2024 - Text-Video Retrieval',description="""<h2>Made by`UTE-AI`</h2>""")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clip_model =  CLIPModel(device = device)
search_clip = SearchFaiss(
                    bin_path = CLIP_BIN,
                    json_path = CLIP_JSON,
                    encoder_model= clip_model)

blip2_model =  BLIP2Model(device = device)
search_blip2 = SearchFaiss(
                    bin_path = BLIP2_BIN,
                    json_path = BLIP2_JSON,
                    encoder_model= blip2_model)

internvideo2_model = InternVideo2Model(device = device)
search_internvideo2 = SearchFaiss(
                    bin_path = INTERNVIDEO2_BIN,
                    json_path = INTERNVIDEO2_JSON,
                    encoder_model= internvideo2_model)

ocr_model = BKAIModel(device=device)
search_ocr_bkai = SearchFaiss(
                bin_path=OCR_BIN,
                json_path=OCR_JSON,
                encoder_model=ocr_model)
search_ocr_ctrl_f = CtrlFSearch(txt_file = OCR_TXT)

dict_img2ytb = read_json_file(YTB_DICT_JSON)
imgreward_method = ImageRewardMethod(device)
vote_method = VotedMethod(device, search_internvideo2)
class UserRequest(BaseModel):
    k: int
    text: str
    rerank: str
    mode_search: str
   
class Results:
    scores: List[np.ndarray]
    idx_images: List[int]
    image_paths: List[str]     
    
model_dict = {
    "clip": search_clip,
    "internvideo2": search_internvideo2,
    "blip2": search_blip2,
    "ocr_bkai": search_ocr_bkai,
    "ocr_ctrlf": search_ocr_ctrl_f
}

rerank_dict = {
    "None": None,
    "ImgReward": imgreward_method,
    "Vote": vote_method
}

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/text_search')
async def text_search(request: UserRequest):
    print("Text ", request.text, "K ", request.k)
    
    search_method = model_dict.get(request.mode_search)
    rerank_method = rerank_dict.get(request.rerank)
    scores, images_id, image_paths = search_method.search_query(request.text, request.k, rerank=rerank_method)
    
    ytb_url = list(map(lambda image_item: dict_img2ytb[image_item], image_paths))

    # search_method.save_result(image_paths, save_path = "./results")
    
    results = {'image_paths': image_paths,
               'url': ytb_url}
    return JSONResponse(content=jsonable_encoder(results), status_code=200)
        
    
# if __name__ == "__main__":
#     uvicorn.run('api:app', host='0.0.0.0', port=8090, reload=False, workers=Pool()._processes)