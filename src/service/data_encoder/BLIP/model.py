import torch
from PIL import Image
from pathlib import Path
import sys

from lavis.models import load_model_and_preprocess

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from src.abstraction.encoder_model import EncoderModel

class BLIP_Model(EncoderModel):
    def __init__(self, device: str):
        self.device = device
        self.__model, self.__vis_processors, self.__text_processors = self.load_model()
        
    def load_model(self):
        model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large",
                                                                           device=self.device, is_eval=True)
        return model, vis_processors, text_processors
    
    def text_encoder(self, text: str):
        
        pass