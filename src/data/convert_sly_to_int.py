import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import hydra
import numpy as np
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from supervisely import Polygon
from tqdm import tqdm

from src.data.utils import CLASS_ID, METADATA_COLUMNS

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_mask_properties(
    figure_bitmap: dict,
    mask: np.ndarray,
) -> Tuple[str, Polygon, List[List[Any]]]:
    mask = mask.astype(bool)
    bitmap = figure_bitmap['data']
    mask_ = sly.Bitmap.base64_2_data(bitmap)
    mask[
        figure_bitmap['origin'][1] : figure_bitmap['origin'][1] + mask_.shape[0],
        figure_bitmap['origin'][0] : figure_bitmap['origin'][0] + mask_.shape[1],
    ] = mask_[:, :]

    encoded_mask = sly.Bitmap.data_2_base64(mask)
    mask = sly.Bitmap(mask)
    contour = mask.to_contours()[0]
    bbox = [
        [min(contour.exterior_np[:, 1]), min(contour.exterior_np[:, 0])],
        [max(contour.exterior_np[:, 1]), max(contour.exterior_np[:, 0])],
    ]

    return encoded_mask, contour, bbox


def get_mask_point(
    point: List[int],
    mask: np.ndarray,
) -> Tuple[str, Polygon, List[List[Any]]]:
    mask = cv2.circle(mask, point, 30, 255, -1)
    mask = mask.astype(bool)
    encoded_mask = sly.Bitmap.data_2_base64(mask)
    mask = sly.Bitmap(mask)
    contour = mask.to_contours()[0]
    bbox = [
        [min(contour.exterior_np[:, 1]), min(contour.exterior_np[:, 0])],
        [max(contour.exterior_np[:, 1]), max(contour.exterior_np[:, 0])],
    ]

    return encoded_mask, contour, bbox


def get_object_coords(
    obj: dict,
) -> dict:
    """Extract box coordinates from a Supervisely annotation.

    Args:
        obj: dictionary with information about one object from supervisely annotations
    Returns:
        dictionary which contains coordinates for a rectangle (left, top, right, bottom)
    """
    if obj['geometryType'] == 'bitmap':
        bitmap = sly.Bitmap.base64_2_data(obj['bitmap']['data'])
        x1, y1 = obj['bitmap']['origin'][0], obj['bitmap']['origin'][1]
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
    box_height = abs(y2 - y1 + 1)
    box_width = abs(x2 - x1 + 1)

    return box_height, box_width


# TODO: use mask instead of obj
def get_object_area(
    obj: dict,
) -> int:
    area = 5

    return area


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
        area_label = 'Small'
    elif 32 * 32 <= area <= 96 * 96:
        area_label = 'Medium'
    else:
        area_label = 'Large'

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
            xy = get_object_coords(obj)
            get_box_size(*xy.values())

            # TODO: add extraction of additional columns
            obj_info = {
                'image_path': dst_img_path,
                'image_name': img_name,
                'slide': slide,
                'tile': tile,
                'dataset': study,
                'image_width': img_width,
                'image_height': img_height,
                'class_id': CLASS_ID[class_name],
                'class': class_name,
            }
            print(obj_info)

            if obj_type == 'bitmap' or obj_type == 'point':
                mask = np.zeros((img_height, img_width))
                encoded_mask, contour, bbox = (
                    get_mask_properties(
                        figure_bitmap=obj['bitmap'],
                        mask=mask,
                    )
                    if obj['geometryType'] == 'bitmap'
                    else get_mask_point(mask=mask, point=obj['points']['exterior'][0])
                )
                if encoded_mask is not None:
                    result_dict = {
                        'image_path': dst_img_path,
                        'image_name': img_name,
                        'slide': slide,
                        'tile': tile,
                        'dataset': study,
                        'image_width': img_width,
                        'image_height': img_height,
                        'x1': bbox[0][0],
                        'y1': bbox[0][1],
                        'x2': bbox[1][0],
                        'y2': bbox[1][1],
                        'box_width': bbox[1][0] - bbox[0][0],
                        'box_height': bbox[1][1] - bbox[0][1],
                        'box_area': '',
                        'box_label': '',
                        'area': int(contour.area),
                        'mask': encoded_mask,
                        'class_id': CLASS_ID[obj['classTitle']],
                        'class': obj['classTitle'],
                    }
                    df_ann = df_ann.append(result_dict, ignore_index=True)
                else:
                    pass
            else:
                log.warning('Annotation ObjectType unknown')
    return df_ann


def annotation_parsing(
    project: sly.Project,
    save_dir: str,
):
    annotation = Parallel(n_jobs=1, backend='threading')(  # TODO: set to -1
        delayed(parse_single_annotation)(
            dataset=dataset,
            save_dir=save_dir,
        )
        for dataset in tqdm(project, desc='Annotation parsing')
    )

    return annotation


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_sly_to_int',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    project = sly.Project(cfg.data_dir, sly.OpenMode.READ)

    # 1. Annotation parsing
    df_list = annotation_parsing(
        project=project,
        save_dir=cfg.save_dir,
    )

    # 2. Save annotation data frame
    df = pd.concat(df_list)
    df.sort_values(['image_path', 'class_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    save_path = os.path.join(cfg.save_dir, 'metadata.xlsx')
    df.to_excel(
        save_path,
        sheet_name='Metadata',
        index=True,
        index_label='id',
    )


if __name__ == '__main__':
    main()
