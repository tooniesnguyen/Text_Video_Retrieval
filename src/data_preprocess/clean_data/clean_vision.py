import os
import glob
import sys
from cleanvision import Imagelab

from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from src.utils.logger import register
logger = register.get_tracking("clean_data.clean_vision.py")

import argparse
####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='Clean Vision')
parser.add_argument('--input_dir', default='/home/toonies/Learn/Text_Video_Retrieval/data/images/Keyframes_L02', type=str, help= None)
args = parser.parse_args()
####################################################################

def remove_images_not_in_keep_list(all_images, keep_images):
    for image_path in all_images.copy():  # Tạo một bản sao của danh sách để tránh lỗi khi loại bỏ phần tử
        if image_path not in keep_images:
            try:
                os.remove(image_path)
                logger.info(f"Deleted image: {image_path}")
                all_images.remove(image_path)  # Loại bỏ khỏi danh sách tất cả các ảnh
            except OSError as e:
                logger.info(f"Error deleting image {image_path}: {e}")

def list_subdirectories(path):
    subdirectories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return subdirectories


def main():
    directory_path = args.input_dir
    subdirectories = list_subdirectories(directory_path)

    for subdir in subdirectories:
        path = os.path.join(directory_path, subdir)
        logger.info(f"Current path: {path}")
        # Specify path to folder containing the image files in your dataset
        imagelab = Imagelab(data_path=path)
        # Automatically check for a predefined list of issues within your dataset
        imagelab.find_issues()
        # Produce a neat report of the issues found in your dataset
        imagelab.report()
        image_clean = imagelab.issues.query("is_near_duplicates_issue")
        check = image_clean.index.tolist()[::2]
        remove_images_not_in_keep_list(image_clean.index.tolist(), check)

        # Double check :))
        imagelab = Imagelab(data_path=path)
        # Automatically check for a predefined list of issues within your dataset
        imagelab.find_issues()
        # Produce a neat report of the issues found in your dataset
        imagelab.report()
        image_clean = imagelab.issues.query("is_near_duplicates_issue")
        check = image_clean.index.tolist()[::2]
        remove_images_not_in_keep_list(image_clean.index.tolist(), check)


if __name__ == "__main__":
    main()
