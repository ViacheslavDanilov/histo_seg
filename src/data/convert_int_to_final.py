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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import MaskProcessor
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


def split_dataset(
    df: pd.DataFrame,
    train_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    slides = df.slide.unique()
    train_slides, test_slides = train_test_split(
        slides,
        train_size=train_size,
        shuffle=True,
        random_state=seed,
    )

    df_train = df[df['slide'].isin(train_slides)]
    df_test = df[df['slide'].isin(test_slides)]

    return df_train, df_test


def save_metadata(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    save_dir: str,
) -> None:
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train.loc[:, 'split'] = 'train'
    df_test.loc[:, 'split'] = 'test'
    df = pd.concat([df_train, df_test], ignore_index=True)
    df.drop(columns=['id'], inplace=True)
    df.sort_values(['image_path', 'class_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    save_path = os.path.join(save_dir, 'metadata.csv')
    df.to_csv(save_path, index_label='id')


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_int_to_final',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    for subset in ['train', 'test']:
        for dir_type in ['img', 'mask', 'mask_color']:
            os.makedirs(f'{cfg.save_dir}/{subset}/{dir_type}', exist_ok=True)

    # Read and process metadata
    df_path = os.path.join(cfg.data_dir, 'metadata.csv')
    df = pd.read_csv(df_path)
    df_filtered = process_metadata(df=df, classes=cfg.class_names)

    # Split dataset
    df_train, df_test = split_dataset(
        df=df_filtered,
        train_size=cfg.train_size,
        seed=cfg.seed,
    )

    # Save metadata
    save_metadata(
        df_train=df_train,
        df_test=df_test,
        save_dir=cfg.save_dir,
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
            save_dir=f'{cfg.save_dir}/train',
        )
        for img_path, df in tqdm(gb_train, desc='Process train subset')
    )

    Parallel(n_jobs=-1, backend='threading')(
        delayed(process_mask)(
            img_path=img_path,
            df=df,
            smooth_mask=cfg.smooth_mask,
            save_dir=f'{cfg.save_dir}/test',
        )
        for img_path, df in tqdm(gb_test, desc='Process test subset')
    )


if __name__ == '__main__':
    main()
