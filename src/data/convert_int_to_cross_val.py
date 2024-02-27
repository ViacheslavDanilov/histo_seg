import logging
import os
from typing import List, Tuple

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.data.convert_int_to_final import process_mask, process_metadata, save_metadata

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def cross_validation_split(
    df: pd.DataFrame,
    id_column: str,
    num_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    ids = df[id_column].unique()
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=11)

    splits = []
    for train_idx, val_idx in kf.split(ids):
        splits.append(
            (
                df[df[id_column].isin(train_idx)],
                df[df[id_column].isin(val_idx)],
            ),
        )

    return splits


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_int_to_cross_val',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    for _ in range(cfg.split_count):
        for subset in ['train', 'test']:
            for dir_type in ['img', 'mask', 'mask_color']:
                os.makedirs(f'{cfg.save_dir}/split_{_}/{subset}/{dir_type}', exist_ok=True)

    # Read and process metadata
    df_path = os.path.join(cfg.data_dir, 'metadata.csv')
    df = pd.read_csv(df_path)
    df_filtered = process_metadata(df=df, classes=cfg.class_names)

    # Split dataset
    splits = cross_validation_split(
        df=df_filtered,
        id_column=cfg.split_column,
        num_folds=cfg.split_count,
    )

    for _, (df_train, df_test) in enumerate(splits):
        # Save metadata
        save_metadata(
            df_train=df_train,
            df_test=df_test,
            save_dir=f'{cfg.save_dir}/split_{_}',
        )

        gb_train = df_train.groupby('image_path')
        gb_test = df_test.groupby('image_path')
        log.info(f'Train images...: {len(gb_train)}')
        log.info(f'Test images....: {len(gb_test)}')

        # Process train and test subsets
        Parallel(n_jobs=-1, backend='threading')(
            delayed(process_mask)(
                img_path=img_path,
                df=df,
                smooth_mask=cfg.smooth_mask,
                save_dir=f'{cfg.save_dir}/split_{_}/train',
            )
            for img_path, df in tqdm(gb_train, desc='Process train subset')
        )

        Parallel(n_jobs=-1, backend='threading')(
            delayed(process_mask)(
                img_path=img_path,
                df=df,
                smooth_mask=cfg.smooth_mask,
                save_dir=f'{cfg.save_dir}/split_{_}/test',
            )
            for img_path, df in tqdm(gb_test, desc='Process test subset')
        )


if __name__ == '__main__':
    main()
