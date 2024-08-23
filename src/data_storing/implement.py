import argparse
from faiss_storing import FaissDB



####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='CLIP Encoder')
parser.add_argument('--i', default='/home/toonies/Learn/Text_Video_Retrieval/data/dicts/npy_clip', type=str, help= "Input Dir")
parser.add_argument('--o', default='/home/toonies/Learn/Text_Video_Retrieval/data/dicts', type=str, help= "Output Dir")
args = parser.parse_args()
####################################################################



if __name__ == "__main__":
    npy_path = args.i
    bin_path = args.o
    clip_model = FaissDB("CLIP",512,"cosine")
    clip_model.convert_npy2bin(npy_path, bin_path)