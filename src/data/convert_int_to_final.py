import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.utils import CLASS_COLOR, CLASS_ID, get_figure_to_mask

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_mask(
    img_path: str,
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    # TODO 1: drop rows excluded classes in process_metadata
    # TODO 2: Too complicated condition. I think it is better to process dataframe before this step
    if (
        len(df) > 0 and len(list(set(class_names) & set(df.class_name.unique()))) > 0
    ):  # TODO: remove condition
        image_width = int(df.image_width.unique())
        image_height = int(df.image_height.unique())
        mask = np.zeros((image_height, image_width), dtype='uint8')
        mask_color = np.zeros((image_height, image_width, 3), dtype='uint8')
        mask_color[:, :] = (128, 128, 128)
        for _, row in df.iterrows():
            if row.class_name in class_names:  # TODO: remove condition
                try:
                    figure_data = sly.Bitmap.base64_2_data(row.encoded_mask).astype(int)
                    mask = get_figure_to_mask(
                        mask=mask,
                        figure=figure_data,
                        class_id=CLASS_ID[row.class_name],
                        points_start=[row['x1'], row['y1']],
                        points_end=[row['x2'], row['y2']],
                    )
                    mask_color[mask == CLASS_ID[row.class_name]] = CLASS_COLOR[row.class_name]
                except Exception as error:
                    print(error)

        img_name = Path(img_path).name
        new_img_path = os.path.join(save_dir, 'img', img_name)
        mask_path = os.path.join(save_dir, 'mask', img_name)
        color_mask_path = os.path.join(save_dir, 'mask_color', img_name)
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(color_mask_path, mask_color)
        shutil.copy(img_path, new_img_path)


def process_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df_out = df.drop('id', axis=1)
    # TODO: filter data by classes, drop empty masks, etc.

    return df_out


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


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_int_to_final',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    for subset in ['train', 'val']:
        for dir_type in ['img', 'mask', 'mask_color']:
            os.makedirs(f'{cfg.save_dir}/{subset}/{dir_type}', exist_ok=True)

    # Read and process metadata
    df_path = os.path.join(cfg.data_dir, 'metadata.xlsx')
    df = pd.read_excel(df_path)
    df_filtered = process_metadata(df)
    # TODO: call process_metadata

    # Split dataset
    df_train, df_test = split_dataset(
        df=df_filtered,
        train_size=cfg.train_size,
        seed=cfg.seed,
    )
    gb_train = df_train.groupby(['image_path'])
    gb_test = df_test.groupby(['image_path'])
    log.info(f'Train images...: {len(gb_train)}')
    log.info(f'Test images....: {len(gb_test)}')

    # Process train and test subsets
    Parallel(n_jobs=-1, backend='threading')(
        delayed(process_mask)(
            img_path=img_path,
            df=df,
            save_dir=f'{cfg.save_dir}/train',
        )
        for img_path, df in tqdm(gb_train, desc='Preparing the training subset')
    )

    Parallel(n_jobs=-1, backend='threading')(
        delayed(process_mask)(
            img_path=img_path,
            df=df,
            save_dir=f'{cfg.save_dir}/test',
        )
        for img_path, df in tqdm(gb_test, desc='Preparing the testing subset')
    )


if __name__ == '__main__':
    main()
