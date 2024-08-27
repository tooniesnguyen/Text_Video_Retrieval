from pathlib import Path
import sys
import os
import glob
from tqdm import tqdm
import faiss
import re
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from src.abstraction.store_db import StoreDB
from src.utils.logger import register
logger = register.get_tracking("Autoshot.implement.py")


class FaissDB(StoreDB):
    def __init__(self, model_name, feat_shape = 512, method = "cosine", *args):
        self.model_name = model_name
        self.method = method
        if self.method in 'L2':
            self.index = faiss.IndexFlatL2(feat_shape)
        elif self.method in 'cosine':
            self.index = faiss.IndexFlatIP(feat_shape)
        else:
            assert f"{method} not supported"

    def __sort_key(self,file_path):
        file_name = os.path.basename(file_path)
        match = re.match(r'(\D+)(\d+)_(V)(\d+)', file_name)
        if match:
            prefix = match.group(1)
            number = int(match.group(2))
            suffix_number = int(match.group(4))
            return (prefix, number, suffix_number)
        return (file_name, 0, 0)
    
    def merge_npy2bin(self, npy_path: str, bin_path: str):
        npy_files = glob.glob(os.path.join(npy_path, "*.npy"))
        npy_files_sorted = sorted(npy_files, key=self.__sort_key)
        for npy_file in npy_files_sorted:
            feats = np.load(npy_file)
            self.index.add(feats)
        
        logger.info(f"Saved at: faiss_{self.model_name}_{self.method}.bin")
        faiss.write_index(self.index, os.path.join(bin_path, f"faiss_{self.model_name}_{self.method}.bin"))
        
        return None    
