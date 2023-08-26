import json
import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import CLASS_ID, METADATA_COLUMNS

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_obj_coords(
    obj: dict,
) -> dict:
    """Extract box coordinates from a Supervisely annotation.

    Args:
        obj: dictionary with information about an object from Supervisely annotations
    Returns:
        dictionary which contains coordinates for a rectangle (left, top, right, bottom)
    """
    if obj['geometryType'] == 'bitmap':
        bitmap = sly.Bitmap.base64_2_data(obj['bitmap']['data'])
        x1 = obj['bitmap']['origin'][0]
        y1 = obj['bitmap']['origin'][1]
        x2 = x1 + bitmap.shape[1]
        y2 = y1 + bitmap.shape[0]
    else:
        xs = [x[0] for x in obj['points']['exterior']]
        ys = [x[1] for x in obj['points']['exterior']]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    return {
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
    }


def get_box_size(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Tuple[int, int]:
    """Extract box sizes by its coordinates.

    Args:
        x1: left x
        y1: top y
        x2: right x
        y2: bottom y
    Returns:
        width and height of a box
    """
    box_height = abs(y2 - y1)
    box_width = abs(x2 - x1)

    return box_height, box_width


def get_object_area(
    obj: dict,
) -> Tuple[int, str]:
    """Use an encoded mask to get the object area and its label.

    Args:
        obj: dictionary with information about an object from Supervisely annotations

    Returns:
        area: the area of an object and its label
    """
    if obj['geometryType'] == 'bitmap':
        encoded_mask = obj['bitmap']['data']
        mask = sly.Bitmap.base64_2_data(encoded_mask)
        area = np.count_nonzero(mask)
    elif obj['geometryType'] == 'point':
        area = 1
    else:
        raise ValueError(f'Unknown geometry type: {obj["geometryType"]}')

    label = get_area_label(area)

    return area, label


def get_area_label(
    area: int,
) -> str:
    """Extract box sizes by its coordinates.

    Args:
        area: object area
    Returns:
        area label of an object
    """
    if area < 32 * 32:
        area_label = 'small'
    elif 32 * 32 <= area <= 96 * 96:
        area_label = 'medium'
    else:
        area_label = 'large'

    return area_label


def parse_single_annotation(
    dataset: sly.Dataset,
    save_dir: str,
) -> pd.DataFrame:
    df_ann = pd.DataFrame(columns=METADATA_COLUMNS)
    study = dataset.name
    img_dir = os.path.join(save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    for img_name in dataset:
        ann_path = os.path.join(dataset.ann_dir, f'{img_name}.json')
        ann = json.load(open(ann_path))
        src_img_path = os.path.join(dataset.img_dir, img_name)
        dst_img_path = os.path.join(img_dir, img_name)
        shutil.copy(src_img_path, dst_img_path)

        slide, tile = Path(img_name).stem.split('_')
        img_height = ann['size']['height']
        img_width = ann['size']['width']

        for obj in ann['objects']:
            obj_type = obj['geometryType']
            class_name = obj['classTitle']
            xy = get_obj_coords(obj)
            box_height, box_width = get_box_size(*xy.values())
            area, area_label = get_object_area(obj)
            encoded_mask = obj['bitmap']['data'] if obj_type == 'bitmap' else ''

            obj_info = {
                'image_path': dst_img_path,
                'image_name': img_name,
                'slide': slide,
                'tile': tile,
                'dataset': study,
                'image_width': img_width,
                'image_height': img_height,
                'type': obj_type,
                'x1': xy['x1'],
                'y1': xy['y1'],
                'x2': xy['x2'],
                'y2': xy['y2'],
                'box_width': box_width,
                'box_height': box_height,
                'area': area,
                'area_label': area_label,
                'encoded_mask': encoded_mask,
                'class_id': CLASS_ID[class_name],
                'class_name': class_name,
            }

            df_ann = df_ann.append(obj_info, ignore_index=True)

    return df_ann


def save_metadata(
    df: pd.DataFrame,
    save_dir: str,
) -> None:
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    save_path = os.path.join(save_dir, 'metadata.csv')
    df.to_csv(
        save_path,
        index=True,
        index_label='id',
    )


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_sly_to_int',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    project = sly.Project(cfg.data_dir, sly.OpenMode.READ)

    # Parse Supervisely dataset and get metadata
    df_list = Parallel(n_jobs=-1, backend='threading')(
        delayed(parse_single_annotation)(
            dataset=dataset,
            save_dir=cfg.save_dir,
        )
        for dataset in tqdm(project, desc='Supervisely dataset processing')
    )

    # Save metadata
    df = pd.concat(df_list)
    df.sort_values(['image_path', 'class_id'], inplace=True)
    save_metadata(
        df=df,
        save_dir=cfg.save_dir,
    )


if __name__ == '__main__':
    main()
