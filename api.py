import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.service.data_retrieval.faiss_search import SearchFaiss
from src.service.data_encoder.CLIP import CLIPModel
from src.service.data_encoder.BLIP2 import BLIP2Model
from src.service.data_encoder.InternVideo2 import InternVideo2Model
from src.utils.utils import load_yaml
from src.utils.config import *

import os
from multiprocessing import Pool
app = FastAPI()



clip_model =  CLIPModel(device = device)

search = SearchFaiss(bin_path = CLIP_BIN,
                    json_path = CLIP_JSON,
                        encoder_model= clip_model)

class UserRequest(BaseModel):
    text: str
    image_id: int
    image_path: str
    k: int
    
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/text_retrieval')
async def text_retrieval(request: UserRequest):
    print("Text ", request.text, "K ", request.k)
    images_path = search.search_query(request.text, request.k, rerank=False)
    
    results = {'image_paths': images_path}
    return JSONResponse(content=jsonable_encoder(results), status_code=200)
    
    

if __name__ == "__main__":
    uvicorn.run('api:app', host='0.0.0.0', port=8090, reload=True, workers=Pool()._processes)