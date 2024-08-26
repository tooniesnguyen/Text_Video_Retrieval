import numpy as np
import os
import io
import cv2

import torch

from pathlib import Path
import sys

from numba import int16
from tqdm import tqdm

FILE = Path(__file__).resolve()
print(FILE)
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
print(str(ROOT))


from utils.config_impl import (Config, eval_dict_leaf)
from utils.utils_impl import (retrieve_text, _frame_from_video, setup_internvideo2)
import warnings

import glob

def _frames_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

video = cv2.VideoCapture('/home/nhi/HCMAI/Text_Video_Retrieval/data/dideo/Keyframes_L01/L01_V001.mp4')


def read_txt(txt_path):
    ranges_list = []
    # Open and read the file
    with open(txt_path, 'r') as file:
        for line in file:
            stripped_line = line.strip().strip('[]')
            numbers = list(map(int, stripped_line.split()))
            ranges_list.append(numbers)
    return ranges_list

def extract_segment_frames_from_txt_file(video_frames, txt_path):
    segment_index_list = read_txt(txt_path)
    print("len segment_index_list: ",len(segment_index_list))
    segment_frames_list = [video_frames[i[0]: i[1]] for i in segment_index_list ]
    print("len segment_frames_list: ",len(segment_frames_list))
    return segment_frames_list, segment_index_list

def segment_images_by_segment_index(images_path_list, segment_index_list):

    images_index_array = np.array([x.split('/')[-1].replace('.jpg', '') for x in images_path_list]).astype(int)
    images_path_list = np.array(images_path_list)
    print(images_index_array)
    segment_images_list = []
    for segment_index in segment_index_list:
        segment_images_list.append(images_path_list[np.where((images_index_array >= segment_index[0]) & (images_index_array <= segment_index[1]))])

    # print(segment_index_list)
    # print(segment_images_list)
    return segment_images_list




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


class InternVideo2Feats:
    def __init__(self, device:str):
        self.device = device
        self.config = Config.from_file("utils/internvideo2_stage2_config.py")
        config = eval_dict_leaf(self.config)

        model_pth = "weights"
        self.config['pretrained_path'] = model_pth

        intern_model, self.tokenizer = setup_internvideo2(config)
        self.vlm = intern_model.to(device)  #MODEL
        pass
    def create_npy(self, input_videos_path, input_images_path, output_npy_dir):
        """create_npy files.
             Args:
                 input_videos_path (str): The input videos path which contains: Keyframe_L01, Keyframe_L02,....
                 input_images_path (str): The input images path which contains: Keyframe_L01, Keyframe_L02,....
                 output_npy_dir (str): The output of .npy files
        """

        fn = self.config.get('num_frames', 8)
        size_t = self.config.get('size_t', 224)

        if os.path.exists(input_videos_path) and os.path.exists(input_images_path):
            video_paths = tqdm(os.listdir(input_videos_path))

            for keyframe_videos_path in video_paths:
                print(keyframe_videos_path)
                for video_path in glob.glob(os.path.join(input_videos_path, keyframe_videos_path) + "/*.mp4"):
                    print(video_path)

                    txt_path = video_path.replace(input_videos_path, input_images_path).replace("mp4", "txt")
                    if os.path.exists(txt_path):
                        images_path = txt_path.replace(".txt", "")  #.../L01_V001, .../L01_V002, ....
                        images_path = glob.glob(f'{images_path}/*.jpg')
                        images_path_list = sorted(images_path, key=lambda x: x.split('/')[-1].replace('.jpg', ''))  # Sắp xếp theo thứ tự số của hình
                        # print(images_path_list)
                        # print("images_path: ", images_path)

                        video = cv2.VideoCapture(video_path)
                        video_frames = [x for x in _frames_from_video(video)]

                        segment_frames_list, segment_index_list = extract_segment_frames_from_txt_file(video_frames, txt_path)
                        segment_images_list = segment_images_by_segment_index(images_path_list, segment_index_list)


                        re_feats = []
                        for segment_frames, segment_images, segment_index in zip(segment_frames_list, segment_images_list, segment_index_list):
                            frames_tensor = frames2tensor(segment_frames, fnum=fn, target_size=(size_t, size_t), device=self.device)
                            vid_feat = self.vlm.get_vid_feat(frames_tensor)
                            # label_probs = (100.0 * vid_feat @ txt_feat.T).softmax(dim=-1)
                            vid_feat /= vid_feat.norm(dim=-1, keepdim=True)
                            vid_feat = vid_feat.detach().cpu().numpy().astype(np.float16).flatten()
                            for image_id in segment_images:
                                re_feats.append(vid_feat)

                        npy_dir = os.path.join(output_npy_dir, txt_path.split("/")[-1].replace(".txt", ''))
                        np.save(npy_dir, re_feats)
                            # print(f"segment_index: {segment_index} ---", vid_feat.shape)



                    else:
                        warnings.warn(f"Can not find directory: {txt_path}")
        else:
            print("ERROR: input_videos_path or input_images_path is not exist")
            return



        pass
    def create_bin(self, input_dir, output_dir):
        pass

def main():
    input_videos_path = "/home/nhi/HCMAI/Text_Video_Retrieval/data/dideo"
    input_images_path = "/home/nhi/HCMAI/Text_Video_Retrieval/data/frames"
    output_npy_dir = "/home/nhi/HCMAI/Text_Video_Retrieval/data/dicts/npy_Internvideo2"
    _InternVideo2Feats = InternVideo2Feats(device="cuda")
    _InternVideo2Feats.create_npy(input_videos_path, input_images_path, output_npy_dir)

if __name__ == "__main__":
    main()