import os
from pathlib import Path
from src.utils.utils import load_yaml


FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]

DATA_DIR = "/media/hoangtv/New Volume/backup/data_aic2024"
CONFIG_FILE = 'config/config.yaml'

device = "cuda:1"
data = load_yaml(path=os.path.join(WORK_DIR,CONFIG_FILE))

DATA_DICT_JSON = os.path.join(DATA_DIR,data['data']['dict_json'])
DATA_DICT_CSV= os.path.join(DATA_DIR,data['data']['dict_csv'])
YTB_DICT_JSON = os.path.join(WORK_DIR, data['data']['ytb_file'])

CLIP_JSON = os.path.join(WORK_DIR,data['clip']['path_json'])
CLIP_BIN = os.path.join(WORK_DIR,data['clip']['path_bin'])

BLIP2_JSON = os.path.join(WORK_DIR,data['blip2']['path_json'])
BLIP2_BIN = os.path.join(WORK_DIR,data['blip2']['path_bin'])

INTERNVIDEO2_JSON = os.path.join(WORK_DIR,data['internvideo2']['path_json'])
INTERNVIDEO2_BIN = os.path.join(WORK_DIR,data['internvideo2']['path_bin'])

OCR_JSON = os.path.join(WORK_DIR,data['ocr']['path_json'])
OCR_BIN = os.path.join(WORK_DIR,data['ocr']['path_bin'])
OCR_TXT = os.path.join(WORK_DIR,data['ocr']['path_txt'])

ASR_JSON = os.path.join(WORK_DIR,data['asr']['path_json'])
ASR_BIN = os.path.join(WORK_DIR,data['asr']['path_bin'])
ASR_TXT = os.path.join(WORK_DIR,data['asr']['path_txt'])