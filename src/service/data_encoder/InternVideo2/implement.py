import numpy as np
import os
import io
import cv2
import torch

# Corrected import paths
from utils.config_impl import (Config, eval_dict_leaf)
from utils.utils_impl import (retrieve_text, _frame_from_video, setup_internvideo2)
print("Hello World1")

# Load video and process frames
video = cv2.VideoCapture('/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/src/service/data_encoder/InternVideo2/utils/example1.mp4')
frames = [x for x in _frame_from_video(video)]

# Sample text candidates for video retrieval
text_candidates = ["A boy is playing with a dog.", "A child is playing with a cat."]

# Load and configure the model
config = Config.from_file('/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/src/service/data_encoder/InternVideo2/utils/internvideo2_stage2_config.py')
print("Hello World12")

config = eval_dict_leaf(config)
print("Hello World1")

model_pth = 'src/data_encoder/Internvideo2/weights/InternVideo2-stage2_1b-224p-f4.pt'
config['pretrained_path'] = model_pth

# Setup InternVideo2 model and tokenizer
intern_model, tokenizer = setup_internvideo2(config)

# Retrieve text descriptions from frames
texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=1, config=config)

# Display results
for t, p in zip(texts, probs):
    print(f'text: {t} ~ prob: {p:.4f}')

