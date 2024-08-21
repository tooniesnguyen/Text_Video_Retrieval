from src.abstraction.encoder_model import EncoderModel
from PIL import Image
import clip
import torch
import numpy as np

class CLIP_Model(EncoderModel):
    def __init__(self, device: str) -> None:
        self.device = device
        self.model, self.preprocess = self.load_model()
        
    def load_model(self):
        return clip.load("ViT-B/32", device=self.device)

    def text_encoder(self, text: str):
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(text).cpu().detach().numpy().astype(np.float32)
        return text_feat

    def image_encoder(self, image_path: str):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feat = self.model.encoder_image(image)
            
        image_feat /= image_feat.norm(dim=-1, keepdim=True)
        image_feat = image_feat.detach().cpu().numpy().astype(np.float16).flatten() 
        
        return image_feat