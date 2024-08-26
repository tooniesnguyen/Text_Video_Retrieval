import argparse
from faiss_storing import FaissDB

from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    

####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='CLIP Encoder')
parser.add_argument('--i', default='/home/toonies/Learn/Text_Video_Retrieval/data/dicts/npy/clip', type=str, help= "Input Dir")
parser.add_argument('--o', default='/home/toonies/Learn/Text_Video_Retrieval/data/dicts', type=str, help= "Output Dir")
args = parser.parse_args()
####################################################################



if __name__ == "__main__":
    npy_path = args.i
    bin_path = args.o
    store_db = FaissDB("CLIP",512,"cosine")
    store_db.merge_npy2bin(npy_path, bin_path)