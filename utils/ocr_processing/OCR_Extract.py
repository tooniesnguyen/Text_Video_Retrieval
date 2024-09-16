from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from craft_text_detector import Craft
import requests
import json
import os
from PIL import Image
from pyvi import ViTokenizer
import cv2
import numpy as np
from underthesea import text_normalize,classify
from unidecode import unidecode
import re
import tqdm
from nltk.corpus import stopwords
import torch
import time
import os
import argparse
import glob
import warnings
warnings.filterwarnings("ignore")

torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=True
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False


detector = Predictor(config)

craft = Craft(output_dir=None, crop_type="box", cuda=config['device'] != 'cpu')

"""
python OCR_Extract.py 
"""

####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--root_path', default='/content/keyframes', type=str, help='folder path to input images')
parser.add_argument('--save_path', default='/content/Ocr', type=str, help='folder path to input images')
args = parser.parse_args()
####################################################################

def ocr_img(img_path: str):
    frame = cv2.imread(img_path)
    prediction_result = craft.detect_text(frame)
    boxes = prediction_result['boxes']
    if len(boxes) > 0:
        boxes = sorted(boxes, key = lambda x:x[0][1])
    output = []
    for idx, box in enumerate(boxes):
        try:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            point1, point2, point3, point4 = box
            x, y, w, h = point1[0], point1[1], point2[0] - point1[0], point4[1]-point1[1]
            crop_img = frame[y:y+h, x:x+w]
            crop_img = Image.fromarray(crop_img)
            s = detector.predict(crop_img)
            output.append(s)
        
        except:
            output.append('')

    text = ' '.join(output)
    text = text.lower()
    return text

root_path = args.save_path  # Đường dẫn đến thư mục lưu kết quả
root_images = args.root_path  # Đường dẫn đến thư mục 'images'
keyframes_paths = sorted(glob.glob(os.path.join(root_images, "Keyframes_*")))

print("Keyframes paths:", keyframes_paths)
print("\nWill save in path", root_path)

for keyframes_path in keyframes_paths:
    print("Running path:", keyframes_path)
    keyframes_name = os.path.basename(keyframes_path)
    os.makedirs(os.path.join(root_path, keyframes_name), exist_ok=True)
    
    with open(os.path.join(root_path, keyframes_name, f"{keyframes_name}.txt"), 'w') as p:
        video_folders = sorted(glob.glob(os.path.join(keyframes_path, "L*")))
        
        for video_folder in tqdm.tqdm(video_folders):
            print(video_folder)
            start_time = time.time()

            chars_to_replace = "^*()-_+=,\"\'?%#@!~$^&|;<>{}[]"
            video_folder_name = os.path.basename(video_folder)

            img_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))

            with open(os.path.join(root_path, keyframes_name, f"{video_folder_name}.txt"), 'w') as f:
                for img_path in img_paths:
                    content_img = ocr_img(img_path)
                    if content_img not in ["", " "]:
                        content_img = re.sub(f"[{re.escape(chars_to_replace)}]", "", content_img)
                        content_txt = [keyframes_name, os.path.splitext(os.path.basename(img_path))[0]]
                        content_txt.append(content_img)
                        list_to_str = ','.join(str(s) for s in content_txt)
                        f.write(list_to_str + "\n")
                        p.write(list_to_str + "\n")

                print("Saved path", os.path.join(root_path, keyframes_name, f"{video_folder_name}.txt"))
            print("elapsed time : {}s".format(time.time() - start_time))


