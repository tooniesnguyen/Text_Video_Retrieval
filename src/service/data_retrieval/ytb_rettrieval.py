import os
import json
import pandas as pd

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

if __name__ == "__main__":
    print(retrieval_video("images/Keyframes_L01/L01_V014/016486.jp"))
    