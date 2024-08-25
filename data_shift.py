import pandas as pd
import numpy as np
import tqdm
import os
import cv2

from get_probability import compute_probability
from psi import PSI
from constants import COLUMNS



class CSFilter:
    
    def __init__(
        self,
        threshold: float = 0.2,
        is_remove: bool = False
    ) -> None:
        self.threshold = threshold
        self.metric = PSI()
        self.is_remove = is_remove
        
        
    def filter(
        self,
        keyframes_dir: str,
        path_result: str
    ) -> None:
        
        drift_scores = []
        drift_status = []
        current_paths = []
        next_paths = []
        final_status = []
        
        os.makedirs(path_result, exist_ok=True)
        
        for name_keyframes in tqdm.tqdm(os.listdir(keyframes_dir), colour='green', desc='Filter by Covariate Shift Detection'):
            
            path_keyframes = os.path.join(keyframes_dir, name_keyframes)
            
            for sub_keyframes in os.listdir(path_keyframes):
                
                image_dir = os.path.join(path_keyframes, sub_keyframes)
                image_paths = os.listdir(image_dir)
                
                for i in range(len(image_paths)-1):
                    
                    current_path = os.path.join(image_dir, image_paths[i])
                    next_path = os.path.join(image_dir, image_paths[i+1])
                    
                    current_image = cv2.imread(current_path)
                    next_image = cv2.imread(next_path)

                    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB).reshape(-1, 1)
                    next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB).reshape(-1, 1)
                    
                    df_current = pd.DataFrame(current_image, columns=COLUMNS)
                    df_next = pd.DataFrame(next_image, columns=COLUMNS)
                    
                    for col in COLUMNS:
                    
                        ref, pro = compute_probability(df_current[col], df_next[col])
                        score = self.metric.run(ref, pro)
                        status = 1 if score > self.threshold else 0
                        
                        drift_scores.append(score)
                        drift_status.append(status)
                        
                    if self.is_remove and all(drift_status):
                        os.remove(next_path)
                    
                    current_paths.append(current_path)
                    next_paths.append(next_path)
                    final_status.append(1 if all(drift_status) else 0)
                    
                df_results = pd.DataFrame({
                    'current_path': current_paths,
                    'next_path': next_paths,
                    'red_score': drift_scores[0::2],
                    'green_score': drift_scores[1::2],
                    'blue_score': drift_scores[2::2],
                    'status': final_status
                })
                
                df_results.to_csv(os.path.join(path_result, f'{sub_keyframes}.csv'))

        
                    
                        

                    
                    
                