import numpy as np
# import os
# import io
import cv2

import torch

from pathlib import Path
import sys
# from numba import int16
# from tqdm import tqdm
# import warnings
# import glob
# import faiss
# import re
# import argparse


FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
print(ROOT)

from src.service.data_encoder.InternVideo2.utils.config_impl import (Config, eval_dict_leaf)
from src.service.data_encoder.InternVideo2.utils.utils_impl import (retrieve_text, _frame_from_video, setup_internvideo2)
from src.abstraction.encoder_model import EncoderModel
# from src.utils.logger import register

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data/255.0-v_mean)/v_std

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

class EncoderInternVideo2(EncoderModel):
    def __init__(self, device: str, *args) -> None:
        self.device = device
        self.__model, self.__config = self.load_model()

    def load_model(self):

        print("Loading Config")
        config = Config.from_file("/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/src/service/data_encoder/InternVideo2/utils/internvideo2_stage2_config.py")
        config = eval_dict_leaf(config)
        model_pth = "/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/src/service/data_encoder/InternVideo2/weights/InternVideo2-stage2_1b-224p-f4.pt"
        config['pretrained_path'] = model_pth

        print("Loading Model")
        intern_model, self.tokenizer = setup_internvideo2(config)
        vlm = intern_model.to(self.device)  # MODEL

        return vlm, config


    def text_encoder(self, text: str):
        text_feat = self.__model.get_txt_feat(text).cpu().detach().numpy().astype(np.float32)
        return text_feat

    def image_encoder(self, image_path: str):
        fn = self.__config.get('num_frames', 8)
        size_t = self.__config.get('size_t', 224)
        video_path = image_path
        video = cv2.VideoCapture(video_path)
        video_frames = [x for x in _frame_from_video(video)]

        frames_tensor = frames2tensor(video_frames, fnum=fn, target_size=(size_t, size_t), device=self.device)
        vid_feat = self.__model.get_vid_feat(frames_tensor)

        # vid_feat /= vid_feat.norm(dim=-1, keepdim=True)
        vid_feat = vid_feat.detach().cpu().numpy().astype(np.float16).flatten()

        return vid_feat
