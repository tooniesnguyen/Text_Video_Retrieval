import os
import numpy as np
import pandas as pd
import torch
import argparse
from PIL import Image
import numpy as np
import glob
import os
import pandas as pd
import glob
import json
import clip
import torch
import faiss
import json
import matplotlib.pyplot as plt
import math
import re
import time
from langdetect import detect
import googletrans
import translate


##lit ra chuồng gà :)))
class Translation:
    def __init__(self, from_lang='vi', to_lang='en', mode='googletrans'):
        # The class Translation is a wrapper for the two translation libraries, googletrans and translate.
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang

        if mode in 'googletrans':
            self.translator = googletrans.Translator()
        elif mode in 'translate':
            self.translator = translate.Translator(from_lang=from_lang,to_lang=to_lang)

    def preprocessing(self, text):
        """
        It takes a string as input, and returns a string with all the letters in lowercase
        :param text: The text to be processed
        :return: The text is being returned in lowercase.
        """
        return text.lower()

    def __call__(self, text):
        """
        The function takes in a text and preprocesses it before translation
        :param text: The text to be translated
        :return: The translated text.
        """
        text = self.preprocessing(text)
        return self.translator.translate(text) if self.__mode in 'translate' \
                else self.translator.translate(text, dest=self.__to_lang).text

translate = Translation()
device = "cuda:0"
print(device)
model, preprocess = clip.load("ViT-B/16", device=device)

def process_ocr_files(ocr_inf, ocr_save_np):
    # Đọc dữ liệu từ tệp OCR vào DataFrame
    df_ocr = pd.read_csv(ocr_inf, delimiter=",", header=None)

    # Biến theo dõi độ dài chuỗi dài nhất
    max_len = 0

    # Duyệt qua từng đường dẫn (unique) trong cột đầu tiên của df_ocr
    for path in df_ocr[0].unique():
        # Trích xuất tên của thư mục cuối cùng trong đường dẫn
        folder_name = os.path.basename(path)
        
        # Xác định đường dẫn lưu trữ kết quả
        save_path = os.path.join(ocr_save_np, folder_name)

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(save_path, exist_ok=True)

        re_feats = []  # Danh sách lưu trữ các đặc trưng của văn bản

        # Duyệt qua từng hàng trong df_ocr tương ứng với đường dẫn đang xét
        for i in range(len(df_ocr.loc[df_ocr[0] == path])):
            # Dịch văn bản từ cột thứ 3
            text = translate(df_ocr[2][i])

            # Cập nhật độ dài tối đa của chuỗi văn bản
            if len(text) >= max_len:
                max_len = len(text)
            print("max len of str is ", max_len)

            # Mã hóa văn bản thành tensor
            text = clip.tokenize([text]).to(device)

            # Mã hóa văn bản thành đặc trưng tensor bằng mô hình đã định nghĩa
            with torch.no_grad():
                text_features = model.encode_text(text).to(device)

            # Chuyển tensor thành numpy array và lưu vào danh sách
            text_features = text_features.cpu().detach().numpy().astype(np.float32)
            re_feats.append(text_features)

        # Lưu danh sách đặc trưng thành tệp .npy
        outfile = f'{save_path}/{folder_name}.npy'
        np.save(outfile, re_feats)
        print(f"Saved {outfile}")

    # Thông báo số lượng mẫu đã được xử lý
    print(f"Processed {len(df_ocr)} samples")

if __name__ == "__main__":
    # Tạo bộ phân tích đối số
    parser = argparse.ArgumentParser(description="Process OCR data and save features as .npy files.")

    # Thêm đối số cho đường dẫn tệp OCR và thư mục lưu trữ
    parser.add_argument('--ocr_inf', type=str, required=True, help="Path to the OCR information file.")
    parser.add_argument('--ocr_save_np', type=str, required=True, help="Directory path to save .npy files.")

    # Phân tích đối số từ dòng lệnh
    args = parser.parse_args()

    # Gọi hàm xử lý với các đối số đã phân tích
    process_ocr_files(args.ocr_inf, args.ocr_save_np)
