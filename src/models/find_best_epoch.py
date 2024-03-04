import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import PROJECT_DIR

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='find_best_epoch',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    metrics_path = os.path.join(PROJECT_DIR, cfg.metrics_path)
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)

    df = pd.read_excel(metrics_path)
    gb = df.groupby(['Model', 'Fold'])

    # Initialize an empty list to store DataFrame objects
    data_frames = []

    # for model_name in tqdm(model_names, desc='Process model', unit=' models'):
    for (model_name, fold), df_sample in tqdm(gb, desc='Process folds', unit=' folds'):
        # Filter rows for training data
        df_test = df_sample[(df_sample['Split'] == 'test') & (df_sample['Class'] == 'Mean')]

        # Get the optimal epoch and its metrics
        max_f1_index = df_test['F1'].idxmax()
        best_epoch = df_test.loc[max_f1_index, 'Epoch']

        # Filter rows for testing data on the best epoch
        df_best_epoch_test = df_sample[
            (df['Epoch'] == best_epoch) & (df['Model'] == model_name) & (df['Split'] == 'test')
        ]

        # Pivot the dataframe to have classes as columns and F1 scores as values
        df_fold = df_best_epoch_test.pivot(
            index=['Model', 'Fold', 'Epoch'],
            columns='Class',
            values='F1',
        )

        # Reorder columns according to the specified header order
        header_order = [
            'Arteriole lumen',
            'Arteriole media',
            'Arteriole adventitia',
            'Venule lumen',
            'Venule wall',
            'Capillary lumen',
            'Capillary wall',
            'Immune cells',
            'Nerve trunks',
            'Mean',
        ]
        df_fold = df_fold[header_order]

        # Append DataFrame to the list
        data_frames.append(df_fold)

    # Concatenate all DataFrames
    df_out = pd.concat(data_frames)

    # Sort final dataframe
    df_out = df_out.sort_values(by=['Fold', 'Model'])

    # Save the dataframe to a CSV file
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_metrics.xlsx')
    df_out.to_excel(save_path)

    log.info('Complete')


if __name__ == '__main__':
    main()
