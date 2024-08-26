import numpy as np
import os
import io
import cv2

import torch

from utils.config_impl import (Config,
                    eval_dict_leaf)

from utils.utils_impl import (retrieve_text,_frame_from_video, setup_internvideo2)
# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', force_download=True)


video = cv2.VideoCapture('utils/example1.mp4')
frames = [x for x in _frame_from_video(video)]
text_candidates = ["A boy are playing with a dog.",
                   "A child are playing with a cat.",
                   "Two person are driving.",
                   "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
                   "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
                   "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
                   "A playful dog slides down a snowy hill, wagging its tail with delight.",
                   "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
                   "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
                   "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."]
config = Config.from_file('utils/internvideo2_stage2_config.py')
config = eval_dict_leaf(config)
model_pth = '/weights'
config['pretrained_path'] = model_pth


intern_model, tokenizer = setup_internvideo2(config)

texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=10, config=config)

for t, p in zip(texts, probs):
    print(f'text: {t} ~ prob: {p:.4f}')
    
    
    
print("Hello World")