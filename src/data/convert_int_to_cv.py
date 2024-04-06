import logging
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold
from tqdm import tqdm

from src import PROJECT_DIR
from src.data.mask_processor import MaskProcessor
from src.data.utils import CLASS_COLOR, CLASS_ID, convert_base64_to_numpy

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_mask(
    img_path: str,
    df: pd.DataFrame,
    smooth_mask: bool,
    save_dir: str,
) -> None:
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    mask = np.zeros((img_height, img_width), dtype='uint8')
    mask_color = np.zeros((img_height, img_width, 3), dtype='uint8')
    mask_color[:, :] = (128, 128, 128)
    mask_processor = MaskProcessor()
    for _, row in df.iterrows():
        obj_mask = convert_base64_to_numpy(row.encoded_mask).astype('uint8')
        if smooth_mask:
            obj_mask = mask_processor.smooth_mask(mask=obj_mask)
            obj_mask = mask_processor.remove_artifacts(mask=obj_mask)
        mask = build_mask(
            mask=mask,
            obj_mask=obj_mask,
            class_id=CLASS_ID[row.class_name],  # type: ignore
            origin=[row['x1'], row['y1']],
        )
        mask_color[mask == CLASS_ID[row.class_name]] = CLASS_COLOR[row.class_name]

    img_stem = Path(img_path).stem
    new_img_path = os.path.join(save_dir, 'img', f'{img_stem}.jpg')
    mask_path = os.path.join(save_dir, 'mask', f'{img_stem}.png')
    color_mask_path = os.path.join(save_dir, 'mask_color', f'{img_stem}.png')
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(color_mask_path, mask_color)
    cv2.imwrite(new_img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def build_mask(
    mask: np.ndarray,
    obj_mask: np.ndarray,
    class_id: int,
    origin: List[int],
) -> np.ndarray:
    obj_height, obj_width = obj_mask.shape
    mask_new = np.zeros_like(mask)
    mask_new[
        origin[1] : origin[1] + obj_height,
        origin[0] : origin[0] + obj_width,
    ] = obj_mask[:, :]
    mask[mask_new == 1] = class_id
    return mask


def process_metadata(
    df: pd.DataFrame,
    classes: List[str],
) -> pd.DataFrame:
    df = df[df['class_name'].isin(classes)]
    df = df[df['area'] > 0]
    return df


def update_metadata(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    fold_idx: int,
) -> pd.DataFrame:
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train.loc[:, 'split'] = 'train'
    df_test.loc[:, 'split'] = 'test'
    df_train.loc[:, 'fold'] = fold_idx
    df_test.loc[:, 'fold'] = fold_idx

    df = pd.concat([df_train, df_test], ignore_index=True)
    df.drop(columns=['id', 'image_path', 'encoded_mask', 'type'], inplace=True)

    df.sort_values(['image_name', 'class_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1

    return df


def merge_and_save_metadata(
    dfs: List[pd.DataFrame],
    save_dir: str,
) -> None:
    df_merged = pd.concat(dfs, ignore_index=True)
    df_merged.index += 1

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'metadata.csv')
    df_merged.to_csv(save_path, index_label='id')


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

    dfs = []
    for fold_idx, (df_train, df_test) in enumerate(splits, start=1):
        # Update metadata
        df = update_metadata(
            df_train=df_train,
            df_test=df_test,
            fold_idx=fold_idx,
        )
        dfs.append(df)

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

    # Merge fold dataframes and save as a single CSV file
    merge_and_save_metadata(
        dfs=dfs,
        save_dir=save_dir,
    )

    log.info('Complete')


if __name__ == '__main__':
    main()
