from transformers import AutoModel,AutoTokenizer
import tqdm
import torch
import numpy as np
import pandas as pd
import os

def load_model():
        model = AutoModel.from_pretrained('dangvantuan/vietnamese-embedding-LongContext',trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained('dangvantuan/vietnamese-embedding-LongContext',trust_remote_code=True)
        return model, tokenizer
def __mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def text_encoder( text: str, device):
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        with torch.no_grad():
            model_output = model(input_ids=input_ids, attention_mask=attention_mask)

        text_feat = __mean_pooling(model_output, attention_mask).cpu().detach().numpy().astype(np.float32)
        return text_feat
      
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model, tokenizer = load_model()
model.to(device)



txt_path = r"F:\Ocr_for_AIC\Text_Video_Retrieval\data\info_ocr_filtered.txt"
npy_path = r"F:\Ocr_for_AIC\Text_Video_Retrieval\data\npy_ocr"

df_ocr = pd.read_csv(txt_path, delimiter=",", header=None)

index = 0
os.makedirs(f"{npy_path}", exist_ok = True)
re_feats = []
for i in tqdm.tqdm(range(len(df_ocr[2]))):
    text = df_ocr[2][index]
    embeddings = text_encoder(text, device).reshape(1,-1)
    # print("Shape of text_embeddings ", text_embeddings.shape) # (1, 256)
    index += 1
    re_feats.append(embeddings)
outfile = 'F:\Ocr_for_AIC\Text_Video_Retrieval\data\npy_ocr\output_asr_ocr.npy' # Edit
np.save(outfile, re_feats)
print(f"Save {outfile}")


