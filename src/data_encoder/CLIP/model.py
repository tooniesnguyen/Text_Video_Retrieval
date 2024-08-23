from pathlib import Path
import sys
from PIL import Image
import clip
import torch
import numpy as np
import glob
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from src.abstraction.encoder_model import EncoderModel

from src.abstraction.store_db import StoreDB
from src.utils.logger import register
logger = register.get_tracking("CLIP.implement.py")

class CLIPModel(EncoderModel):
    def __init__(self, device: str) -> None:
        self.device = device
        self.__model, self.__preprocess = self.load_model()
        
    def load_model(self):
        print("Loading model")
        return clip.load("ViT-B/32", device=self.device)

    def text_encoder(self, text: str):
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feat = self.__model.encode_text(text).cpu().detach().numpy().astype(np.float32)
        return text_feat

    def image_encoder(self, image_path: str):
        image = self.__preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feat = self.__model.encode_image(image)
            
        image_feat /= image_feat.norm(dim=-1, keepdim=True)
        image_feat = image_feat.detach().cpu().numpy().astype(np.float16).flatten() 
        
        return image_feat
    
    def convert_image2npy(self, images_path, npy_path):
        video_paths = sorted(glob.glob(f"{images_path}/*/"))
        video_paths = ['/'.join(i.split('/')[:-1]) for i in video_paths]
        
        re_feats = []
        for vd_path in video_paths:
            keyframe_paths = glob.glob(f'{vd_path}/*.jpg')
            keyframe_paths = sorted(keyframe_paths, key=lambda x : x.split('/')[-1].replace('.jpg',''))
            
            for keyframe_path in tqdm(keyframe_paths):
                image_feat = self.image_encoder(keyframe_path)

                re_feats.append(image_feat)
                
        name_npy = video_paths[0].split('/')[-2]
        outfile = f'{npy_path}/{name_npy}.npy'
        np.save(outfile, re_feats)
        logger.info(f"Saved at path {outfile}")
        return None
    
    def convert_text2npy(self, text_file, npy_path):
        pass