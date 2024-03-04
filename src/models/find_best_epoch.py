import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import PROJECT_DIR

log = logging.getLogger(__name__)


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

    for _, df_sample in tqdm(gb, desc='Process folds', unit=' folds'):
        df_test = df_sample[(df_sample['Split'] == 'test') & (df_sample['Class'] == 'Mean')]
        max_f1_index = df_test['F1'].idxmax()
        best_epoch = df_test.loc[max_f1_index, 'Epoch']
        df_best_epoch = df_sample[
            (df_sample['Epoch'] == best_epoch) & (df_sample['Split'] == 'test')
        ]
        df_fold = df_best_epoch.pivot(
            index=['Model', 'Fold', 'Epoch'],
            columns='Class',
            values='F1',
        )
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
        data_frames.append(df_fold)

    df_out = pd.concat(data_frames)
    df_out = df_out.sort_values(by=['Fold', 'Model'])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_metrics.xlsx')
    df_out.to_excel(save_path)

    log.info('Complete')


if __name__ == '__main__':
    main()
