# import cupy as cp
# import numpy as np

# def most_vote(score_model1, result_model1, score_model2, result_model2):
#     score_model1 = cp.array(score_model1)
#     score_model2 = cp.array(score_model2)

#     result_model1 = np.array(result_model1)
#     result_model2 = np.array(result_model2)

#     common_images, indices_model1, indices_model2 = np.intersect1d(result_model1, result_model2, return_indices=True)

#     indices_model1 = np.array(indices_model1)

#     sorted_common_indices = np.argsort(-score_model1.get()[indices_model1])
#     common_images_sorted = common_images[sorted_common_indices]

#     non_common_model1 = np.setdiff1d(result_model1, common_images)
#     non_common_model2 = np.setdiff1d(result_model2, common_images)

#     non_common_model1_sorted = non_common_model1[np.argsort(-score_model1.get()[np.isin(result_model1, non_common_model1)])]
#     non_common_model2_sorted = non_common_model2[np.argsort(-score_model2.get()[np.isin(result_model2, non_common_model2)])]

#     return common_images_sorted.tolist(), non_common_model1_sorted.tolist(), non_common_model2_sorted.tolist()


##################################################################################################################################
# import numpy as np

# def most_vote(score_model1, result_model1, score_model2, result_model2):
#     score_model1 = np.array(score_model1)
#     result_model1 = np.array(result_model1)
#     score_model2 = np.array(score_model2)
#     result_model2 = np.array(result_model2)

#     common_images, indices_model1, indices_model2 = np.intersect1d(result_model1, result_model2, return_indices=True)

#     sorted_common_indices = np.argsort(-score_model1[indices_model1])
#     common_images_sorted = common_images[sorted_common_indices]

#     non_common_model1 = np.setdiff1d(result_model1, common_images)
#     non_common_model2 = np.setdiff1d(result_model2, common_images)

#     non_common_model1_sorted = non_common_model1[np.argsort(-score_model1[np.isin(result_model1, non_common_model1)])]
#     non_common_model2_sorted = non_common_model2[np.argsort(-score_model2[np.isin(result_model2, non_common_model2)])]

#     return common_images_sorted.tolist(), non_common_model1_sorted.tolist(), non_common_model2_sorted.tolist()

# ##################################################################################################################################

def most_vote(score_model1, result_model1, score_model2, result_model2):
    common_images = list(set(result_model1) & set(result_model2))

    common_images.sort(key=lambda x: score_model1[result_model1.index(x)], reverse=True)

    non_common_model1 = [img for img in result_model1 if img not in common_images]
    non_common_model2 = [img for img in result_model2 if img not in common_images]

    # non_common_model1.sort(key=lambda x: score_model1[result_model1.index(x)], reverse=True)
    # non_common_model2.sort(key=lambda x: score_model2[result_model2.index(x)], reverse=True)

    return common_images, non_common_model1, non_common_model2


# Example usage
score_model1 = [0.5, 0.4, 0.1, 0.3, 0.2, 0.1, 0.15, 0.4, 0.1, 0.3, 0.2, 0.1]
result_model1 = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img12.jpg", "img11.jpg", "img22.jpg", "img23.jpg", "img24.jpg", "img25.jpg", "img32.jpg"]

score_model2 = [0.2, 0.1, 0.5, 0.3, 0.9, 1.0, 0.15, 0.4, 0.1, 0.3, 0.2, 0.1]
result_model2 = ["img7.jpg", "img3.jpg", "img1.jpg", "img8.jpg", "img15.jpg", "img14.jpg", "img17.jpg", "img13.jpg", "img11.jpg", "img18.jpg", "img14.jpg", "img14.jpg"]

import time
start_time = time.time()
array_anhchung, array_model1_score_khong_anhchung, array_model2_score_khong_anhchung = most_vote(
    score_model1, result_model1, score_model2, result_model2)

print("Time execute", time.time() - start_time)
print("array_anhchung:", array_anhchung)
print("array_model1_score_khong_anhchung:", array_model1_score_khong_anhchung)
print("array_model2_score_khong_anhchung:", array_model2_score_khong_anhchung)