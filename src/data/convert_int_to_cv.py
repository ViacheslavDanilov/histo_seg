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

from src import PROJECT_DIR
from src.data.convert_int_to_final import process_mask, process_metadata, save_metadata

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def cross_validation_split(
    df: pd.DataFrame,
    id_column: str,
    num_folds: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    ids = df[id_column].unique()
    kf = KFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=seed,
    )
    splits = []
    for train_idx, test_idx in kf.split(ids):
        train_ids = ids[train_idx]
        test_ids = ids[test_idx]
        df_train = df[df[id_column].isin(train_ids)]
        df_test = df[df[id_column].isin(test_ids)]
        splits.append((df_train, df_test))

    return splits


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='convert_int_to_cv',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_dir = os.path.join(PROJECT_DIR, cfg.data_dir)
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)

    for fold_idx in range(cfg.num_folds):
        for subset in ['train', 'test']:
            for dir_type in ['img', 'mask', 'mask_color']:
                os.makedirs(f'{save_dir}/fold_{fold_idx+1}/{subset}/{dir_type}', exist_ok=True)

    # Read and process metadata
    csv_path = os.path.join(data_dir, 'metadata.csv')
    df = pd.read_csv(csv_path)
    df_filtered = process_metadata(df=df, classes=cfg.class_names)

    # Cross-validation split of the dataset
    splits = cross_validation_split(
        df=df_filtered,
        id_column=cfg.split_column,
        num_folds=cfg.num_folds,
        seed=cfg.seed,
    )

    for fold_idx, (df_train, df_test) in enumerate(splits, start=1):
        # Save metadata
        save_metadata(
            df_train=df_train,
            df_test=df_test,
            save_dir=f'{save_dir}/fold_{fold_idx}',
        )

        gb_train = df_train.groupby('image_path')
        gb_test = df_test.groupby('image_path')
        log.info('')
        log.info(f'Fold {fold_idx} - Train images...: {len(gb_train)}')
        log.info(f'Fold {fold_idx} - Test images....: {len(gb_test)}')

        # Process train and test subsets
        Parallel(n_jobs=-1, backend='threading')(
            delayed(process_mask)(
                img_path=img_path,
                df=df,
                smooth_mask=cfg.smooth_mask,
                save_dir=f'{save_dir}/fold_{fold_idx}/train',
            )
            for img_path, df in tqdm(gb_train, desc=f'Process train subset - Fold {fold_idx}')
        )

        Parallel(n_jobs=-1, backend='threading')(
            delayed(process_mask)(
                img_path=img_path,
                df=df,
                smooth_mask=cfg.smooth_mask,
                save_dir=f'{save_dir}/fold_{fold_idx}/test',
            )
            for img_path, df in tqdm(gb_test, desc=f'Process test subset - Fold {fold_idx}')
        )


if __name__ == '__main__':
    main()
