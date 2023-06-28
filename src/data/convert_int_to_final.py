import logging
import os
import shutil
from typing import List

import cv2
import hydra
import numpy as np
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.data.utils import CLASS_ID, CLASS_COLOR, get_figure_to_mask

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_mask(
    img_path: str,
    classes: List[str],
    data: pd.DataFrame,
    save_dir: str,
) -> None:
    if len(data) > 0 and len(list(set(classes) & set(data.class_name.unique()))) > 0:
        mask = np.zeros((int(data.image_width.mean()), int(data.image_height.mean())), dtype='uint8')
        mask_color = np.zeros((int(data.image_width.mean()), int(data.image_height.mean()), 3), dtype='uint8')
        mask_color[:, :] = (128, 128, 128)
        for _, row in data.iterrows():
            if row.class_name in classes:
                try:
                    figure_data = sly.Bitmap.base64_2_data(row.encoded_mask).astype(int)
                    mask = get_figure_to_mask(
                        mask=mask,
                        figure=figure_data,
                        cl_id=CLASS_ID[row.class_name],
                        points_start=[row['x1'], row['y1']],
                        points_end=[row['x2'], row['y2']],
                    )
                    mask_color[mask == CLASS_ID[row.class_name]] = CLASS_COLOR[row.class_name]
                except Exception as error:
                    print(error)

        cv2.imwrite(f'{save_dir}/mask/{os.path.basename(img_path)}', mask)
        cv2.imwrite(f'{save_dir}/mask_color/{os.path.basename(img_path)}', mask_color)
        shutil.copy(img_path, f'{save_dir}/img/{os.path.basename(img_path)}')


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

    df = pd.read_excel(cfg.df_path)
    slides = np.unique(df.slide.values)
    train_slides, test_slides = train_test_split(
        slides,
        train_size=cfg.train_size,
        shuffle=True,
        random_state=cfg.seed,
    )

    train_img_paths = df[df['slide'].isin(train_slides)].image_path.values
    test_img_paths = df[df['slide'].isin(test_slides)].image_path.values
    log.info(f'Train images...: {len(train_img_paths)}')
    log.info(f'Test images....: {len(train_img_paths)}')

    # test_df = df.loc[df['image_path'].isin(test_img_paths)]
    # train_df = df.loc[df['image_path'].isin(train_img_paths)]
    # test_df['class'].value_counts()
    # train_df['class'].value_counts()

    Parallel(n_jobs=-1, backend='threading')(
        delayed(get_mask)(
            img_path=img_path,
            classes=cfg.classes,
            data=df.loc[df.image_path == img_path],
            save_dir=f'{cfg.save_dir}/train',
        )
        for img_path in tqdm(train_img_paths, desc='Preparing the training subset')
    )

    Parallel(n_jobs=-1, backend='threading')(
        delayed(get_mask)(
            img_path=img_path,
            classes=cfg.classes,
            data=df.loc[df.image_path == img_path],
            save_dir=f'{cfg.save_dir}/val',
        )
        for img_path in tqdm(test_img_paths, desc='Preparing the testing subset')
    )


if __name__ == '__main__':
    main()
