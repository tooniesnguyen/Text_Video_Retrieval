import torch
import glob
import os
import argparse
import numpy as np
import torch. nn as nn
from tqdm import tqdm

from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.service.data_preprocess.Autoshot.supernet import TransNetV2Supernet
from src.service.data_preprocess.utils import get_frames, get_batches
from src.service.data_preprocess.utils import predictions_to_scenes 
from src.service.data_preprocess.utils import Result2Text
from src.service.data_preprocess.utils import Result2Image
from src.service.data_preprocess.utils import Visualize2Image
from tqdm import tqdm

from src.utils.logger import register
logger = register.get_tracking("Autoshot.implement.py")


####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='AUTOSHOT')
parser.add_argument('--i', default='/media/hoangtv/New Volume/backup/data_aic2024/videos/Videos_L01', type=str, help= "Input Dir")
parser.add_argument('--o', default='/media/hoangtv/New Volume/backup/data_aic2024/images', type=str, help= "Output Dir")
args = parser.parse_args()
####################################################################


class AutoShotImplement:
    def __init__(self, input_dir: str, output_dir: str):

        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir,os.path.basename(input_dir))
        self.device = "cuda"
        self.model = self.__init_model()

    def __init_model(self):
        model = TransNetV2Supernet().eval()
        pretrained_path = os.path.join(os.path.dirname(__file__),"weights/ckpt_0_200_0.pth")
        logger.info(f"[AutoShot] Using weights from {pretrained_path}.")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrained_path, map_location = self.device)
        pretrained_dict = {k: v for k, v in pretrained_dict['net'].items() if k in model_dict}
        logger.info(f"[AutoShot] Current model has {len(model_dict)} paras, Update paras {len(pretrained_dict)}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = model.to(self.device)
        return model.eval()

    def __predict(self, batch):
        batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]) * 1.0
        batch = batch.to(self.device)
        one_hot = self.model(batch)
        if isinstance(one_hot, tuple):
            one_hot = one_hot[0]
        return torch.sigmoid(one_hot[0])

    def run(self, visualize_result = False) -> None:
        video_paths = sorted(glob.glob(os.path.join(self.input_dir, "*mp4")))
        for video_path in tqdm(video_paths):
            logger.info("[AutoShot] Extracting frames from {}".format(video_path))
            folder_name = video_path.split('/')[-1].replace( '.mp4','')
            folder_path = self.output_dir + f'/{folder_name}'
            os.makedirs(folder_path, exist_ok= True)


            predictions = []
            frames = get_frames(video_path)
            for batch in tqdm(get_batches(frames)):
                one_hot = self.__predict(batch)
                one_hot = one_hot.detach().cpu().numpy()

                predictions.append(one_hot[25:75])
            predictions = np.concatenate(predictions, 0)[:len(frames)]
            scenes = predictions_to_scenes(predictions)
            Result2Text(folder_path, predictions= scenes)
            Result2Image(video_file=video_path, img_dir=folder_path, scenes= scenes)
            if visualize_result:
                Visualize2Image(video_path, scenes, "autoshot")
                visualize_result = False

def main():
    input_dir = args.i
    output_dir = args.o
    model_impl = AutoShotImplement(input_dir, output_dir)
    model_impl.run()

if __name__ == "__main__":
    main()


