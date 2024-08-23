from model import CLIPModel
from pathlib import Path
import sys
import argparse
import os
import glob
from tqdm import tqdm
from PIL import Image


FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from src.abstraction.store_db import StoreDB
from src.utils.logger import register
logger = register.get_tracking("Autoshot.implement.py")


####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='CLIP Encoder')
parser.add_argument('--i', default='/home/toonies/Learn/Text_Video_Retrieval/data/images/Keyframes_L01', type=str, help= "Input Dir")
parser.add_argument('--o', default='/home/toonies/Learn/Text_Video_Retrieval/data/dicts/npy_clip', type=str, help= "Output Dir")
args = parser.parse_args()
####################################################################


class FaissDB(StoreDB):
    def __init__(self, model):
        self.model = model
    
    def convert_image2npy(self, images_path, npy_path):
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
        logger.info(f"Saved at path {outfile}")
        return None
    
    def convert_text2npy(self, text_file, npy_path):
        pass
    def convert_npy2bin(self, bin_path: str, npy_path: str, method = "cosine", feat_shape = 512):
        pass
    
if __name__ == "__main__":
    imgs_path = args.i
    npy_path = args.o
    clip_model = CLIPModel(device="cuda")
    faiss_db = FaissDB(model=clip_model)
    faiss_db.convert_image2npy(images_path=imgs_path, npy_path=npy_path)
        