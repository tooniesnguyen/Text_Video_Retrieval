from pathlib import Path
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.abstraction.encoder_model import EncoderModel
from src.utils.logger import register
logger = register.get_tracking("OCR.implement.py")

class BKAIModel(EncoderModel):
    def __init__(self, device: str, *args) -> None:
        self.device = device
        self.__model, self.__tokenizer = self.load_model()
        self.__model.to(self.device)  # Chuyển mô hình sang thiết bị CUDA

    def load_model(self):
        print("Loading model OCR")
        model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
        tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

        return [model, tokenizer]

    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def text_encoder(self, text: str):
        encoded_input = self.__tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_input['input_ids'].to(self.device)  # Chuyển dữ liệu đầu vào sang thiết bị CUDA
        attention_mask = encoded_input['attention_mask'].to(self.device)  # Chuyển dữ liệu đầu vào sang thiết bị CUDA

        with torch.no_grad():
            model_output = self.__model(input_ids=input_ids, attention_mask=attention_mask)
        
        text_feat = self.__mean_pooling(model_output, attention_mask).cpu().detach().numpy().astype(np.float32)
        return text_feat

    def image_encoder(self, image_path: str):
        pass

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Tự động xác định thiết bị
    bkai_encoder = BKAIModel(device)
    text_feat = bkai_encoder.text_encoder("Hai người đang đánh nha")
    print("Text Encoder: ", text_feat.shape)
