import os
import json
import pandas as pd


import os
import sys
from pathlib import Path
from tqdm import tqdm

FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[3]
if str(WORK_DIR) not in sys.path:
    sys.path.append(str(WORK_DIR))
from src.utils.config import DATA_DICT_JSON, DATA_DICT_CSV

def get_watch_url(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    watch_url = data.get('watch_url')
    return watch_url

def retrieval_video(image_path):
    parts = image_path.split('/')
    keyframe = parts[2] 
    id_frame = parts[3].split('.')[0] 
    file_json = os.path.join(DATA_DICT_JSON, f"{keyframe}.json")
    file_csv = os.path.join(DATA_DICT_CSV, f"{keyframe}.csv")
    fps = pd.read_csv(file_csv)['fps'][0]
    cal_time = int(int(id_frame)/fps)
    url_ytb = get_watch_url(file_json)
    url_time = f"{url_ytb}&t={cal_time}s"
    return url_time

def convert_path2ytb(path_json_image, path_json_ytb):
    with open(path_json_image, 'r') as f:
        image_list = json.load(f)
    result_dict = {image_path: retrieval_video(image_path) for image_path in tqdm(image_list)}
    with open(path_json_ytb, 'w') as o:
        json.dump(result_dict, o, indent=4)

if __name__ == "__main__":
    # print(retrieval_video("images/Keyframes_L01/L01_V014/016486.jp"))
    path_json_image = "/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/data/dicts/json/keyframes_id_search.json"
    path_json_ytb = "/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/data/dicts/json/convert_image2ytb.json"
    convert_path2ytb(path_json_image, path_json_ytb)
        
    