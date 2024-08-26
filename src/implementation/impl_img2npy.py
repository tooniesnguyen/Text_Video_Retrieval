import argparse

from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from src.service.data_encoder.CLIP import CLIPModel
from service.data_storing.feat_storing import StoreFeatureModel

####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='CLIP Encoder')
parser.add_argument('--i', default='/home/toonies/Learn/Text_Video_Retrieval/data/images/Keyframes_L01', type=str, help= "Input Dir")
parser.add_argument('--o', default='/home/toonies/Learn/Text_Video_Retrieval/data/dicts/npy/clip', type=str, help= "Output Dir")
args = parser.parse_args()
####################################################################



if __name__ == "__main__":
    imgs_path = args.i
    npy_path = args.o
    clip_model = CLIPModel(model_name = None ,device="cuda")
    store_feat_clip = StoreFeatureModel(clip_model)
    store_feat_clip.save_image2npy(images_path=imgs_path, npy_path=npy_path)