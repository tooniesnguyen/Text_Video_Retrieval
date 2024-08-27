import os
import numpy as np
import pandas as pd
import torch
import argparse
from PIL import Image
import clip
import faiss
import json
import matplotlib.pyplot as plt
import math
import re
import time
from langdetect import detect
import googletrans
import translate


class Translation:
    def __init__(self, from_lang='vi', to_lang='en', mode='googletrans'):
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang

        if mode in 'googletrans':
            self.translator = googletrans.Translator()
        elif mode in 'translate':
            self.translator = translate.Translator(from_lang=from_lang, to_lang=to_lang)

    def preprocessing(self, text):
        return text.lower()

    def __call__(self, text):
        text = self.preprocessing(text)
        return self.translator.translate(text) if self.__mode in 'translate' \
            else self.translator.translate(text, dest=self.__to_lang).text

translate = Translation()
device = "cuda:0"
print(device)
model, preprocess = clip.load("ViT-B/16", device=device)

def process_ocr_files(ocr_inf, ocr_save_np, combine):
    df_ocr = pd.read_csv(ocr_inf, delimiter=",", header=None)
    max_len = 0
    combined_feats = []  # Danh sách để lưu trữ tất cả đặc trưng nếu combine=True

    for path in df_ocr[0].unique():
        folder_name = os.path.basename(path)
        save_path = os.path.join(ocr_save_np, folder_name)
        os.makedirs(save_path, exist_ok=True)
        re_feats = []

        for i in range(len(df_ocr.loc[df_ocr[0] == path])):
            text = translate(df_ocr[2][i])
            if len(text) >= max_len:
                max_len = len(text)
            print("max len of str is ", max_len)
            text = clip.tokenize([text]).to(device)

            with torch.no_grad():
                text_features = model.encode_text(text).to(device)

            text_features = text_features.cpu().detach().numpy().astype(np.float32)
            re_feats.append(text_features)

        if combine:
            combined_feats.extend(re_feats)
        else:
            outfile = f'{save_path}/{folder_name}.npy'
            np.save(outfile, re_feats)
            print(f"Saved {outfile}")

    if combine:
        # Lưu tất cả đặc trưng vào một file .npy duy nhất
        combined_data = np.concatenate(combined_feats, axis=0)
        np.save(os.path.join(ocr_save_np, 'combined_data.npy'), combined_data)
        print(f"Saved combined features to 'combined_data.npy'")

    print(f"Processed {len(df_ocr)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process OCR data and save features as .npy files.")

    parser.add_argument('--ocr_inf', type=str, required=True, help="Path to the OCR information file.")
    parser.add_argument('--ocr_save_np', type=str, required=True, help="Directory path to save .npy files.")
    parser.add_argument('--combine', action='store_true', help="If set, combine all features into a single .npy file. Otherwise, save each folder as a separate file.")

    args = parser.parse_args()

    process_ocr_files(args.ocr_inf, args.ocr_save_np, args.combine)
