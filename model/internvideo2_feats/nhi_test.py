
import cv2
import os
from tqdm import tqdm
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

def create_npy(input_videos_path, input_images_path, output_npy_dir):
    """create_npy files.
         Args:
             input_videos_path (str): The input videos path which contains: Keyframe_L01, Keyframe_L02,....
             input_images_path (str): The input images path which contains: Keyframe_L01, Keyframe_L02,....
             output_npy_dir (str): The output of .npy files
    """
    if os.path.exists(input_videos_path) and os.path.exists(input_images_path):
        video_paths = tqdm(os.listdir(input_videos_path))
        input_txts_path = input_images_path

        for keyframe_videos_path in video_paths:
            print(keyframe_videos_path)
            for video_path in glob.glob(os.path.join(input_videos_path, keyframe_videos_path) + "/*.mp4"):
                print(video_path)
                video = cv2.VideoCapture(video_path)
                video_frames = [x for x in _frames_from_video(video)]

                txt_path = video_path.replace(input_videos_path, input_images_path).replace("mp4", "txt")
                if os.path.exists(txt_path):
                    print(txt_path)
                    segment_frames_list, segment_index_list = extract_segment_frames_from_txt_file(video_frames, txt_path)


                else:
                    warnings.warn(f"Can not find directory: {txt_path}")
    else:
        print("ERROR: input_videos_path or input_images_path is not exist")
        return

input_videos_path = "/home/nhi/HCMAI/Text_Video_Retrieval/data/dideo"
input_images_path = "/home/nhi/HCMAI/Text_Video_Retrieval/data/frames"
output_npy_dir = ""
create_npy(input_videos_path, input_images_path, output_npy_dir)