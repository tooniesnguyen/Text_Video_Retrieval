from pathlib import Path
import sys
import os
import glob
from tqdm import tqdm
import faiss
import re
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from abstraction.store_feat import StoreFeat
from src.utils.logger import register
logger = register.get_tracking("Extract_Model")

class StoreFeatureModel(StoreFeat):
    def __init__(self, model, *args):
        self.model = model

    def save_image2npy(self, images_path, npy_path):
        video_paths = sorted(glob.glob(f"{images_path}/*/"))
        video_paths = ['/'.join(i.split('/')[:-1]) for i in video_paths]
        
        re_feats = []
        for vd_path in video_paths:
            keyframe_paths = glob.glob(f'{vd_path}/*.jpg')
            keyframe_paths = sorted(keyframe_paths, key=lambda x : x.split('/')[-1].replace('.jpg',''))
            
            for keyframe_path in tqdm(keyframe_paths):
                image_feat = self.model.image_encoder(keyframe_path)

                re_feats.append(image_feat)
                
        name_npy = video_paths[0].split('/')[-2]
        outfile = f'{npy_path}/{name_npy}.npy'
        np.save(outfile, re_feats)
        logger.info(f"Saved at path {outfile}")
        return None
    
    def save_text2npy(self, text_file, npy_path):
        pass