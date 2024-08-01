import os
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path
import cv2
import glob


from src.service.data_preprocess.Transnetv2 import TransNetV2
from src.service.data_preprocess.utils import predictions_to_scenes 
from src.service.data_preprocess.utils import Result2Text
from src.service.data_preprocess.utils import Result2Image
from src.service.data_preprocess.utils import Visualize2Image


class TransNetV2Implement:
    def __init__(self, input_dir: str, output_dir: str):
        super(TransNetV2Implement, self).__init__()

        self.model = TransNetV2()
        self.input_dir = input_dir
        self.output_dir = output_dir
    
    def run(self, visualize_result = True) -> None:
        video_paths = sorted(glob.glob(os.path.join(self.input_dir, "*mp4")))
        for video_path in video_paths:
            folder_name = video_path.split('/')[-1].replace( '.mp4','')
            folder_path = self.output_dir + f'/{folder_name}'
            os.makedirs(folder_path)
            video_frames, single_frame_predictions, all_frame_predictions = self.model.predict_video(video_path)
            scenes = predictions_to_scenes(single_frame_predictions)
            Result2Text(folder_path, predictions= scenes)
            Result2Image(video_file=video_path, img_dir=folder_path, scenes= scenes)
            if visualize_result:
                Visualize2Image(video_path, scenes, "transet")
                visualize_result = False









