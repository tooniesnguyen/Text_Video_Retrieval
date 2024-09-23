import numpy as np
import re
import pandas as pd
import os
from tqdm import tqdm  # Thêm tqdm vào để hiển thị thanh tiến độ

def filter_letter(text: str):
    text_filter = ""
    vowel_letters = "o, ô, ơ, a, â, ă, ê, e, u, ư, i, y,\
    ó, ồ, ớ, á, ấ, ắ, ề, é, ụ, ứ, í, ý, ò, ộ, ờ, à, ầ, ằ, ế, è, ú, ừ, ì, ỳ,\
        õ, ổ, ở, ã, ẫ, ẵ, ễ, ẽ, ũ, ữ, ĩ, ỷ, ỏ, ỗ, ợ, ả, ẩ, ẳ, ể, ẻ, ủ, ử, ỉ, ỹ, ọ, ố, ỡ, ạ, ậ,\
            ặ, ệ, ẹ, ù, ự, ị, ỵ, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0"
    for i in text.split(" "):
        if len(i) <=1 and (i != "y"or i!= "ý" ):
            continue
        elif len(i) == 2:
            num_not_vowel = 0
            for letter in i:
                if letter not in vowel_letters:
                    num_not_vowel +=1
            if num_not_vowel == 2:
                continue # Delete word
            else:
                text_filter += (" "+i)
        elif i == "giay" or i=="giây" or i == "tinchinh" or i == "tinchính" or i == "glay":
            continue
        elif re.search(r'\d{2}:\d{2}', i) is not None:
            continue
        elif "000" in i:
            continue
        else:    
            text_filter += (" "+i)
    return text_filter

def filter_text(text: str):
    time_pattern = r'\d{2}:\d{2}'
    match = re.search(time_pattern, text)
    matches = re.findall(time_pattern, text)
    if matches:
        extracted_characters = match.group()+text[text.find(matches[0]) + len(matches[0]) :]
        text_time = extracted_characters
    else:
        text_time = text
    text_return = filter_letter(text_time)
    return text_return

path_txt = r"F:\Ocr_for_AIC\Text_Video_Retrieval\data\info_ocr_2.txt"
df_txt = pd.read_csv(path_txt, delimiter = ",", header=None)

# Thêm thanh tiến độ khi lọc text
for idx in tqdm(range(len(df_txt[2])), desc="Filtering text"):
    try:
        text_filter = filter_text(df_txt[2][idx])
        df_txt[2][idx] = str(text_filter)
    except:
        df_txt[2][idx] = "nan"
        print("Current indx:", idx, df_txt[2][idx])

# Xóa hàng trống hoặc ít ký tự (<=5) và hiển thị thanh tiến độ
num_del = 0
for i in tqdm(range(len(df_txt[2])), desc="Deleting short rows"):
    if df_txt[2][i] == " " or df_txt[2][i] == "" or len(df_txt[2][i]) <= 5 or df_txt[2][i] == "nan":
        num_del += 1
        df_txt = df_txt.drop(index=i)

print("Num row delete: ", num_del)

# Lưu file đã xử lý
path_save_txt = r"F:\Ocr_for_AIC\Text_Video_Retrieval\data\info_ocr_filtered_2.txt"
df_txt.to_csv(path_save_txt, sep=',', index=False, header=False)
