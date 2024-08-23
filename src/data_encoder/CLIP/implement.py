from model import CLIPModel
import argparse




####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='CLIP Encoder')
parser.add_argument('--i', default='/home/toonies/Learn/Text_Video_Retrieval/data/images/Keyframes_L01', type=str, help= "Input Dir")
parser.add_argument('--o', default='/home/toonies/Learn/Text_Video_Retrieval/data/dicts/npy_clip', type=str, help= "Output Dir")
args = parser.parse_args()
####################################################################



if __name__ == "__main__":
    imgs_path = args.i
    npy_path = args.o
    clip_model = CLIPModel(device="cuda")
    clip_model.convert_image2npy(images_path=imgs_path, npy_path=npy_path)