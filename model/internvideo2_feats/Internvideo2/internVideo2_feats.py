import numpy as np
import os
import io
import cv2

import torch

from pathlib import Path
import sys
from numba import int16
from tqdm import tqdm
import warnings
import glob
import faiss
import re
import argparse

FILE = Path(__file__).resolve()
print(FILE)
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
print(str(ROOT))


from utils.config_impl import (Config, eval_dict_leaf)
from utils.utils_impl import (retrieve_text, _frame_from_video, setup_internvideo2)

def _frames_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

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
    # print(images_index_array)
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

def process_name(name: int):
    return "0"*(6-len(str(name))) + str(name)

def sort_key(file_path):
    file_name = os.path.basename(file_path)
    match = re.match(r'(\D+)(\d+)_(V)(\d+)', file_name)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        suffix_number = int(match.group(4))
        return (prefix, number, suffix_number)
    return (file_name, 0, 0)

class InternVideo2Feats:
    def __init__(self, device:str, active_model=True):

        if active_model:
            self.device = device
            self.config = Config.from_file("utils/internvideo2_stage2_config.py")
            config = eval_dict_leaf(self.config)

            model_pth = "weights"
            self.config['pretrained_path'] = model_pth

            intern_model, self.tokenizer = setup_internvideo2(config)
            self.vlm = intern_model.to(device)  #MODEL

            print("Img2npy mode ----- Load model successfully")
        else:
            print("Npy2bin mode")

    def create_npy(self, input_data_path:str, output_npy_dir):
        """create_npy files.
             Args:
                 input_data_path (str): The input data path which contains: video and images folder
                 output_npy_dir (str): The output of .npy files
        """

        fn = self.config.get('num_frames', 8)
        size_t = self.config.get('size_t', 224)

        input_videos_path = os.path.join(input_data_path, "video")
        input_images_path = os.path.join(input_data_path, "images")

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

                    else:
                        warnings.warn(f"Can not find directory: {txt_path}")
        else:
            print("ERROR: input_videos_path or input_images_path is not exist")
            return


    def create_bin(self, input_npy_path:str, output_bin_path:str, method="cosine", feature_shape=512):
        """create bin files.
             Args:
                 input_npy_path (str): The input folder which contains all .npy files
                 output_bin_path (str): The output of .bin files
        """
        if method in 'L2':
            index = faiss.IndexFlatL2(feature_shape)
        elif method in 'cosine':
            index = faiss.IndexFlatIP(feature_shape)
        else:
            assert f"{method} not supported"

        npy_files = glob.glob(os.path.join(input_npy_path, "*.npy"))

        if len(npy_files)==0:
            assert f"Can not find any .npy file. Please check the input path!"

        npy_files_sorted = sorted(npy_files, key=sort_key)

        for npy_file in tqdm(npy_files_sorted):
            feats = np.load(npy_file).astype(np.float32)
            index.add(feats)

        faiss.write_index(index, os.path.join(output_bin_path, f"faiss_InternVideo2_{method}.bin"))

        print(f'Saved {os.path.join(output_bin_path, f"faiss_InternVideo2_{method}.bin")}')

    # write_bin_file_ocr(bin_path=f"{WORK_DIR}/data/dicts/bin_clip", npy_path=f"{WORK_DIR}/data/dicts/npy_clip")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img2npy", action='store_true')
    parser.add_argument("--npy2bin", action='store_true')
    parser.add_argument('-i', required=True, type=str, help="Input .npy or data/keyframes Dir")
    parser.add_argument('-o', required=True, type=str, help="Output .npy or .bin Dir")
    args = parser.parse_args()

    if args.img2npy and not args.npy2bin:
        _InternVideo2Feats = InternVideo2Feats(device="cuda", active_model=True)
        _InternVideo2Feats.create_npy(input_data_path=args.i, output_npy_dir=args.o)

    if args.npy2bin and not args.img2npy:
        _InternVideo2Feats = InternVideo2Feats(device="cuda", active_model=False)
        _InternVideo2Feats.create_bin(input_npy_path=args.i, output_bin_path=args.o)


if __name__ == "__main__":
    main()