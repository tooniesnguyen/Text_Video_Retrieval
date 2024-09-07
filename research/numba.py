import numpy as np

# Given list of image paths
img_paths = [
    "/media/hoangtv/New Volume/backup/data_aic2024/images/Keyframes_L11/L11_V001/008266.jpg",
    "/media/hoangtv/New Volume/backup/data_aic2024/images/Keyframes_L01/L01_V001/024538.jpg",
    "/media/hoangtv/New Volume/backup/data_aic2024/images/Keyframes_L08/L08_V026/019542.jpg",
    "/media/hoangtv/New Volume/backup/data_aic2024/images/Keyframes_L08/L08_V026/000711.jpg",
    "/media/hoangtv/New Volume/backup/data_aic2024/images/Keyframes_L07/L07_V002/016336.jpg"
]

# Given re-ranking indices
idx_reranking = [2, 4, 1, 3, 0]

# Convert img_paths to a NumPy array
img_paths_np = np.array(img_paths)

# Reorder img_paths using idx_reranking
sorted_img_paths_np = img_paths_np[idx_reranking]

# Convert the result back to a list
sorted_img_paths = sorted_img_paths_np.tolist()

# Print the sorted paths
print(sorted_img_paths)
