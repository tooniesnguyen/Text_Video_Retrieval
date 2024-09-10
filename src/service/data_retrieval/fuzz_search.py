from rapidfuzz import fuzz
import pandas as pd 
import json
import re 
import numpy as np 


def fill_ocr_results(st, list_ocr_results):
  def check(x):
    score = fuzz.token_set_ratio(x.lower(), st.lower())
    if score >= 80:
      if re.search(f'\s{st.lower()}\s',x.lower()):
        return True
      return False
    return False
  
  ocr_1 = list(filter(check, list_ocr_results))
  list_ocr=list(map(lambda x:f'{x.split(",")[0]}/{x.split(",")[1]:0>6}.jpg', ocr_1))
  return list_ocr

def fill_ocr_df(st, df):
  def ocr(x):
    score = fuzz.token_set_ratio(x.lower(), st.lower())
    if score > 80:
      if re.search(f'\s{st.lower()}\s',x.lower()):
        return str(score)
      else:
        return np.nan
    return np.nan
    
  df["score"] = df["ocr"].apply(ocr)
  return df.dropna(subset=["score"]).apply(lambda x: f'{x.video_id}/{x.frame_id:0>6}.jpg',axis=1).values

if __name__ == "__main__":
    with open("/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/data/temp/TXT_FILES/info_ocr_processed.txt", "r", encoding="utf8") as fi:
        list_ocr_results = list(map(lambda x: x.replace("\n",""), fi.readlines()))
    import time
    start_time = time.time()
    list_ocr = fill_ocr_results("mimita tennies giáo văn thị thị thị marating marn satman", list_ocr_results)
    print("Time execute: ", time.time()- start_time)
    print(list_ocr)