import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import PROJECT_DIR
from src.data.utils import get_file_list

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='combine_model_results',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_dir = os.path.join(PROJECT_DIR, cfg.data_dir)
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)

    # Initialize an empty list to store DataFrame objects
    data_frames = []

    # Iterate through files in the directory
    csv_list = get_file_list(src_dirs=data_dir, ext_list='.csv')
    for csv_path in tqdm(csv_list, desc='Merging CSV files', unit=' CSV'):
        # Extract fold number and model name from the filename
        file_stem = Path(csv_path).stem
        model_name, _, fold_idx = file_stem.split('_')

        # Read CSV file
        df = pd.read_csv(csv_path)

        # Add additional columns: Fold, Model, and Loss
        df['Loss'] = 1 - df['Dice']
        df['Fold'] = int(fold_idx)
        df['Model'] = model_name

        # Move the 'Loss' column to the second position
        df.insert(1, 'Loss', df.pop('Loss'))

        # Append DataFrame to the list
        data_frames.append(df)

    # Concatenate all DataFrames
    combined_data = pd.concat(data_frames, ignore_index=True)

    # Sort final dataframe
    combined_data = combined_data.sort_values(by=['Fold', 'Model'])

    # Save combined data
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model_results.xlsx')
    combined_data.to_excel(save_path, index=False)

    log.info('Complete')


if __name__ == '__main__':
    main()
