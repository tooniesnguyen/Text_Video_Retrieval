import os
import numpy as np
import faiss
import re
import pandas as pd
import glob

ocr_inf = "/content/info_ocr.txt" ## nhập đường dẫn file txt đã OCR thu được

# Đọc dữ liệu từ tệp OCR vào DataFrame
df_ocr = pd.read_csv(ocr_inf, delimiter=",", header=None)

def process_name(name: int):
    return "0"*(6-len(str(name))) + str(name)

def write_bin_file_ocr(bin_path: str, image_root_path: str, npy_root_path: str, method='cosine', feature_shape=512): # Edit 512, 768

    id = 0
    if method == 'L2':
        index = faiss.IndexFlatL2(feature_shape)
    elif method == 'cosine':
        index = faiss.IndexFlatIP(feature_shape)
    else:
        raise ValueError(f"{method} not supported")

    # Tìm tất cả các file .npy trong thư mục npy_root_path
    npy_files = glob.glob(f"{npy_root_path}/**/*.npy", recursive=True)
    combine_data = npy_root_path+"/combined_data.npy"
    for npy_file in npy_files:
        # Load npy file
        feats = np.load(combine_data)

        # Lấy đường dẫn thư mục chứa file .npy
        dir_path = os.path.dirname(npy_file)

        # Load danh sách các file ảnh trong thư mục tương ứng
        ids = os.listdir(dir_path)
        ids = sorted(ids, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
        idx = -1
        for _ in range(len(df_ocr[1])):
            # Tạo đường dẫn ảnh
            idx += 1
            #print(idx)
            image_path = f"{image_root_path}/{df_ocr[0][idx]}/{process_name(df_ocr[1][idx])}.jpg"

            # Output ID 1, 2, 3, 4 . if len path == id reset id
            feat = feats[id]
            id = 0 if len(df_ocr.loc[df_ocr[0] == df_ocr[0][46]]) == (id + 1) else id + 1

            print("ID: ", id)

            # Chuyển đổi và reshape feature
            feat = feat.astype(np.float32).reshape(1, -1)
            print("##########################################")
            print(" Feat after reshape: ", feat.shape)
            index.add(feat)

            

    # Lưu FAISS index
    faiss.write_index(index, os.path.join(bin_path, f"faiss_LIT_OCR_{method}.bin"))

    print(f'Saved {os.path.join(bin_path, f"faiss_LIT_OCR_{method}.bin")}')
    print(f"Number of Index: {idx}")

# Ví dụ sử dụng
write_bin_file_ocr(
    bin_path="/content/faiss_output",  # Đường dẫn lưu file faiss
    image_root_path="/content/keyframes",  # Đường dẫn chứa các folder chứa file ảnh
    npy_root_path="/content/npy2",  # Đường dẫn chứa các folder chứa file npy
    method='cosine',
    feature_shape=512
)
