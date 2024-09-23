import numpy as np
import glob
import os
import pandas as pd
import glob
import json
import argparse
import tqdm
import torch
import time
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer




######################################################################
# input: # địa chỉ các file *.txt
         # địa chỉ lưu file npy   

# output: các npy dc lưu vào file

######################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SentenceTransformer('keepitreal/vietnamese-sbert')



def main():

        df_ocr = pd.read_csv(txt_paths, delimiter=",", header=None)
        
        index = 0
        os.makedirs(f"{npy_path}", exist_ok = True)
        re_feats = []
        print( f'{npy_path}/final_ASR.npy')
        for i in tqdm.tqdm(range(len(df_ocr[2]))):
            text = df_ocr[2][index] # anhasd asada s

            # print(f"Idx: {index}")
            embeddings = model.encode(text).reshape(1,-1)


            # print("Shape of text_embeddings ", text_embeddings.shape) # (1, 256)
            index += 1
            re_feats.append(embeddings)
        outfile = f'{npy_path}/final_ASR.npy' # Edit
        np.save(outfile, re_feats)
        print(f"Save {outfile}")




if __name__ == '__main__':
    txt_paths = r"F:\Ocr_for_AIC\Text_Video_Retrieval\data\info_ocr_merged.txt"
    npy_path = r"F:\Ocr_for_AIC\Text_Video_Retrieval\files\ocr\npy\npy_ASR_SBERT"
    main()