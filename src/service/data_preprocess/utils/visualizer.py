from PIL import Image, ImageDraw, ImageFont
import numpy as np
import ffmpeg
from pathlib import Path
import os

FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[4]



def get_frames(fn, width=48, height=27):
    video_stream, err = (
        ffmpeg
        .input(fn)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .run(capture_stdout=True, capture_stderr=True)
    )
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])
    return video

def scenes2zero_one_representation(scenes, n_frames):
    prev_end = 0
    one_hot = np.zeros([n_frames], np.uint64)
    many_hot = np.zeros([n_frames], np.uint64)

    for start, end in scenes:
        for i in range(prev_end, start):
            many_hot[i] = 1
        if not (prev_end == 0 and start == 0):
            one_hot_index = prev_end + (start - prev_end) // 2
            one_hot[one_hot_index] = 1

        prev_end = end

    # if scene ends with transition
    if prev_end + 1 != n_frames:
        for i in range(prev_end, n_frames):
            many_hot[i] = 1

        one_hot_index = prev_end + (n_frames - prev_end) // 2
        one_hot[one_hot_index] = 1

    return one_hot, many_hot

def visualize_predict(frames: np.ndarray, predictions: np.ndarray) -> Image:
    """
    Visualize the result of inference
    
    Parameters:
    -----------
    - `frames` (np.ndarray): contains the video frames.
    - `predictions` (np.ndarray): contains the result of transnet/autoshot prediction

    Returns:
    --------
    - `img` (Image): image includes the result of prediction
    """
    predictions = predictions
    width = 25
    ih, iw, ic = frames.shape[1:]
    pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
    frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])
    predictions = [np.pad(x, (0, pad_with)) for x in predictions]
    height = len(frames) // width
    img = frames.reshape([height, width, ih + 1, iw + len(predictions), ic])
    img_tmp = np.concatenate(np.split(
        np.concatenate(np.split(img, height), axis=2)[0], width
    ), axis=2)[0, :-1]
    img = Image.fromarray(img_tmp)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(f"{FILE.parents[0]}/fonts/AbyssinicaSIL-R.ttf", 12)
    for h in range(height):
        for w in range(width):
            avg_c = img_tmp[h * (ih + 1) + 3 : h * (ih + 1) + 9, w * (iw + 1) : w * (iw + 1)+12, :]
            avg_c = avg_c.sum()
            avg_c /= (3 * 6 * 12)
            n = h * width + w
            draw.text((w * (iw + 1),h * (ih + 1)+3), str(n),
                fill=(255, 0, 255) if avg_c < 128 else (0, 0, 0),font=font)

    for i, pred in enumerate(zip(*predictions)):
        x, y = i % width, i // width
        x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

        # we can visualize multiple predictions per single frame
        for j, p in enumerate(pred):
            color = [0, 0, 0]
            color[1] = 255
            value = np.round(p * (ih - 1))
            if value != 0:
                draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=8)

    return img


def Visualize2Image(video_path, scenes, method):
    frames = get_frames(video_path)
    predictions = scenes2zero_one_representation(scenes.astype(np.int),frames.shape[0])
    img_visualize = visualize_predict(frames, predictions)
    output_dir = os.path.join(WORK_DIR,'reports/compared')
    os.makedirs(output_dir, exist_ok= True)
    img_visualize.save(f'{output_dir}/{method}.png')
