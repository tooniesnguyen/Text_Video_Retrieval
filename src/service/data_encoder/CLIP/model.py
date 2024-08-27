from pathlib import Path
import sys
from PIL import Image
import clip
import torch
import numpy as np
import glob
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from src.abstraction.encoder_model import EncoderModel

from src.utils.logger import register
logger = register.get_tracking("CLIP.implement.py")

class CLIPModel(EncoderModel):
    def __init__(self, device: str, *args) -> None:
        self.device = device
        self.__model, self.__preprocess = self.load_model()
        
    def load_model(self):
        print("Loading model")
        return clip.load("ViT-B/32", device=self.device, jit=False)

    def text_encoder(self, text: str):
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feat = self.__model.encode_text(text).cpu().detach().numpy().astype(np.float32)
        return text_feat

    def image_encoder(self, image_path: str):
        image = self.__preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feat = self.__model.encode_image(image)
            
        image_feat /= image_feat.norm(dim=-1, keepdim=True)
        image_feat = image_feat.detach().cpu().numpy().astype(np.float16).flatten() 
        
        return image_feat


if __name__ == "__main__":
    clip_model = CLIPModel("cuda")
    txt = clip_model.text_encoder("A man with a red hat")
    image =clip_model.image_encoder("/home/toonies/Learn/Text_Video_Retrieval/data/images/Keyframes_L02/L02_V002/000471.jpg")
    print("image shape ", image.shape) # (512, )
    print("text shape ", txt.shape) # (1, 512)