from pathlib import Path
import sys
from PIL import Image
import clip
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from src.abstraction.encoder_model import EncoderModel


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