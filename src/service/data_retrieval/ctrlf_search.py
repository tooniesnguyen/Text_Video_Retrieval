from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.abstraction.search_db import SearchDB
from src.utils.config import WORK_DIR, DATA_DIR
import matplotlib.pyplot as plt
import numpy as np
import math
import os

class CtrlFSearch(SearchDB):
    def __init__(self, txt_file: str):
        self.data = self.load_file(txt_file)
        self.default_nan_img = "images/Keyframes_L00/L00_V000/000000.jpg"
    def load_file(self, txt_file):
        with open(txt_file, 'r', encoding='utf-8') as file:
                return file.readlines()
    
    def _extract_keyframe_path(self, line):
        """
        Trích xuất đường dẫn file từ dòng dữ liệu.
        """
        try:
            parts = line.split(',')
            if len(parts) >= 2:
                keyframe_id = parts[0]
                frame_number = parts[1].strip()
                return f"images/Keyframes_{keyframe_id.split('_')[0]}/{keyframe_id}/{int(frame_number):06}.jpg"
            return None
        except Exception as e:
            print(f"Đã xảy ra lỗi khi trích xuất đường dẫn: {e}")
            
    def save_result(self, image_paths, save_path: str):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))

        black_image = np.full((256, 256, 3),255)

        for i in range(1, columns * rows + 1):
            try:
                img = plt.imread(os.path.join(DATA_DIR, image_paths[i - 1])) #
            except:
                img = black_image
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))
            plt.imshow(img)
            plt.axis("off") 

            # Save each subplot as an individual image file in the "./results" directory
        output_path = os.path.join(save_path, f"ctrl_f_search.png")
        plt.savefig(output_path)
        plt.close(fig)

    def search_query(self, text: str, k: int, **kwargs):
        score, idx_image = [0, 0]
        results = [self._extract_keyframe_path(line) for line in self.data if text in line]
        num_to_add = k - len(results)
        if num_to_add < 0:
            num_to_add = 0
        results.extend([self.default_nan_img] * num_to_add)
        
        return score, idx_image, results

if __name__ == "__main__":
    ctrlf_search = CtrlFSearch("/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/data/temp/TXT_FILES/info_ocr_processed.txt")
    _, _, images_path = ctrlf_search.search_query("cảnh báo mưa lớn lũ lên nhanh tại các tỉnh miền trung", 42)
    ctrlf_search.save_result(images_path, "./")