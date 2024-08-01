import numpy as np
from src.utils.logger import Logger, register
import cv2


logger = register.get_tracking(__name__)

def predictions_to_scenes(predictions: np.ndarray, 
                            threshold: float = 0.5) -> np.ndarray:
    
    predictions = (predictions > threshold).astype(np.uint8)
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # just fix if all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)

def Result2Text(txt_dir: str, predictions: np.ndarray) -> None:
    """
    Write results to txt file
    
    Parameters:
    -----------
    - `txt_file` (str): path of txt file. Ex: "./*txt"
    - `predictions` (np.ndarray): contains the result of transnet/autoshot prediction

    Returns:
    --------
    - None
    """
    with open(f'{txt_dir}.txt',"w") as f:
        for predict in predictions:
            f.write(str(predict) + '\n')
    return None
def Result2Image(video_file: str, img_dir: str, scenes: np.ndarray) -> None:
    """
    Write results to txt file
    
    Parameters:
    -----------
    - `txt_file` (str): path of txt file. Ex: "./*txt".
    - `img_dir` (str): dir to save image
    - `scenes` (np.ndarray): contains the result of transnet/autoshot prediction

    Returns:
    --------
    - None
    """
    cam = cv2.VideoCapture(video_file)
    currentframe = 0
    index = 0

    while True:
        ret,frame = cam.read()
        if ret:
            currentframe += 1
            # for sc in scenes:
            if (index>len(scenes)-1):
                break
            idx_first = int(scenes[index][0])
            idx_end = int(scenes[index][1])
            idx_025 = int(scenes[index][0] + (scenes[index][1]-scenes[index][0])/4)
            idx_05 = int(scenes[index][0] + (scenes[index][1]-scenes[index][0])/2)
            idx_075 = int(scenes[index][0] + 3*(scenes[index][1]-scenes[index][0])/4)

            #### First ####
            if currentframe - 1 == idx_first:
                filename_first = "{}/{:0>6d}.jpg".format(img_dir, idx_first)
                # video_save = cv2.resize(video[idx_first], (1280,720))
                cv2.imwrite(filename_first, frame)

            # #### End ####
            if currentframe - 1 == idx_end:
                filename_end = "{}/{:0>6d}.jpg".format(img_dir, idx_end)
                # video_save = cv2.resize(video[idx_end], (1280,720))
                cv2.imwrite(filename_end, frame)
                index += 1

            #### 025 ####
            if currentframe - 1 == idx_025:
                filename_025 = "{}/{:0>6d}.jpg".format(img_dir, idx_025)
                # video_save = cv2.resize(video[idx_025], (1280,720))
                cv2.imwrite(filename_025, frame)

            # #### 05 ####
            if currentframe - 1 == idx_05:
                filename_05 = "{}/{:0>6d}.jpg".format(img_dir, idx_05)
                # video_save = cv2.resize(video[idx_05], (1280,720))
                cv2.imwrite(filename_05, frame)

            # #### 075 ####
            if currentframe - 1 == idx_075:
                filename_075 = "{}/{:0>6d}.jpg".format(img_dir, idx_075)
                # video_save = cv2.resize(video[idx_075], (1280,720))
                cv2.imwrite(filename_075, frame)

        else:
            break

    cam.release()
    cv2.destroyAllWindows()
    logger.info(f"Saved all images on {img_dir}")

def get_batches(frames):
    reminder = 50 - len(frames) % 50
    if reminder == 50:
        reminder = 0
    frames = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25), 0)

    def func():
        for i in range(0, len(frames) - 50, 50):
            yield frames[i:i + 100]

    return func()