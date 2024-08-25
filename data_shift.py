import pandas as pd
import polars as pl
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
        threshold: float = 0.2
    ) -> None:
        self.threshold = threshold
        self.metric = PSI()
        
        
    def filter(
        self,
        keyframes_dir: str,
        path_result: str
    ) -> None:
        
        os.makedirs(path_result, exist_ok=True)
        
        for name_keyframes in tqdm.tqdm(sorted(os.listdir(keyframes_dir)), colour='green', desc='Filter by Covariate Shift Detection'):
            
            path_keyframes = os.path.join(keyframes_dir, name_keyframes)
            
            for sub_keyframes in sorted(os.listdir(path_keyframes)):
                
                drift_scores = []
                current_paths = []
                next_paths = []
                
                image_dir = os.path.join(path_keyframes, sub_keyframes)
                image_paths = sorted(os.listdir(image_dir))
                
                for i in range(len(image_paths)-1):
                    
                    current_path = os.path.join(image_dir, image_paths[i])
                    next_path = os.path.join(image_dir, image_paths[i+1])
                    
                    current_image = cv2.imread(current_path)
                    next_image = cv2.imread(next_path)

                    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
                    next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
                    
                    df_current = pd.DataFrame(current_image, columns=COLUMNS)
                    df_next = pd.DataFrame(next_image, columns=COLUMNS)
                    
                    for col in COLUMNS:
                    
                        ref, pro = compute_probability(df_current[col], df_next[col])
                        score = self.metric.run(ref, pro)
                        status = 1 if score > self.threshold else 0
                        
                        drift_scores.append(score)
                    
                    current_paths.append(current_path)
                    next_paths.append(next_path)
                
                df_results = pl.DataFrame({
                    'current_path': current_paths,
                    'next_path': next_paths,
                    'red_score': drift_scores[0::3],
                    'green_score': drift_scores[1::3],
                    'blue_score': drift_scores[2::3]
                })
                
                df_results = df_results.with_columns(
                    drift_status=pl.when(
                        (pl.col('red_score') > self.threshold)
                        | (pl.col('green_score') > self.threshold)
                        | (pl.col('blue_score') > self.threshold)
                    ).then(1)
                    .otherwise(0)
                )
                
                df_results.write_csv(os.path.join(path_result, f'{sub_keyframes}.csv'))

        
                    
                        

                    
                    
                