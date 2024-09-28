import polars as pl
import os
import glob
import tqdm
import argparse


def parse_arg() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=str, help="Path to the folder containing CSV files")
    return parser.parse_args()


def shuffle(folder_path: str) -> None:
    
    folder_path_info = folder_path.split('/')
    
    folder_save = f'./shuffled_{folder_path_info[-1]}'
    os.makedirs(folder_save, exist_ok=True)
    
    csv_files = sorted(glob.glob(f'{folder_path}/*.csv'))

    if len(csv_files) == 0:
        print(f"No CSV files found in the folder: {folder_path}")
        return

    for file in tqdm.tqdm(csv_files, colour='green', desc='Shuffling the row of csv files'):
        
        name_file = file.split('/')[-1]
        type_challenge = os.path.splitext(name_file)[0].split('-')[-1]
 
        name_columns = ['id_video', 'id_frame']
        dtypes=[pl.Utf8, pl.Utf8]
        
        if type_challenge == 'qa':
            name_columns = ['id_video', 'id_frame', 'qa']
            dtypes=[pl.Utf8, pl.Utf8, pl.Utf8]
            
        schema_overrides= dict(zip(name_columns, dtypes))

        df = pl.scan_csv(
            file, 
            with_column_names=lambda names: name_columns, 
            has_header=False,
            truncate_ragged_lines=True,
            schema_overrides=schema_overrides
        ).collect()
        
        # Drop duplicates
        df = df.drop_nulls()
        df = df.unique()
 
        shuffled_id_video = (
            df
            .select(
                pl.col('id_frame').shuffle(seed=1)
            )
        )
 
        shuffled_df = shuffled_id_video.join(
            df,
            on='id_frame', 
            how='left'
        )
        
        shuffled_df = shuffled_df.select(
            pl.col(name_columns)
        )

        shuffled_df.write_csv(os.path.join(folder_save, name_file), include_header=False)    


if __name__ == "__main__":
    
    args = parse_arg()
    shuffle(args.folder_path)
