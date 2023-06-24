import json
import logging
import multiprocessing
import os
import shutil
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

from src.data.utils import CLASS_ID

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


def parse_single_annotation(
    dataset: sly.VideoDataset,
    save_dir: str,
) -> pd.DataFrame:
    df_ann = pd.DataFrame()
    study = dataset.name

    if not os.path.exists(f'{save_dir}/{study}'):
        os.makedirs(f'{save_dir}/{study}')

    for img_name in dataset:
        ann_path = os.path.join(dataset.ann_dir, f'{img_name}.json')
        ann = json.load(open(ann_path))
        shutil.copy(os.path.join(dataset.img_dir, img_name), f'{save_dir}/{study}/{img_name}')

        slide, tile = img_name.split('.')[0].split('_')

        for obj in ann['objects']:
            if obj['geometryType'] == 'bitmap' or obj['geometryType'] == 'point':
                mask = np.zeros((ann['size']['width'], ann['size']['height']))
                encoded_mask, contour, bbox = get_mask_properties(
                    figure_bitmap=obj['bitmap'],
                    mask=mask,
                ) if obj['geometryType'] == 'bitmap' else get_mask_point(mask=mask, point=obj['points']['exterior'][0])
                if encoded_mask is not None:
                    result_dict = {
                        'image_path': f'{save_dir}/{study}/{img_name}',
                        'image_name': img_name,
                        'slide': slide,
                        'tile': tile,
                        'dataset': study,
                        'image_width': ann['size']['width'],
                        'image_height': ann['size']['height'],
                        'x1': bbox[0][0],
                        'y1': bbox[0][1],
                        'x2': bbox[1][0],
                        'y2': bbox[1][1],
                        'xc': int(np.mean([bbox[0][0], bbox[1][0]])),
                        'yc': int(np.mean([bbox[0][1], bbox[1][1]])),
                        'box_width': bbox[1][0] - bbox[0][0],
                        'box_height': bbox[1][1] - bbox[0][1],
                        'box_area': '',
                        'box_label': '',
                        'area': int(contour.area),
                        'mask': encoded_mask,
                        'class_id': CLASS_ID[obj['classTitle']],
                        'class':  obj['classTitle']
                    }
                    df_ann = pd.concat([df_ann, pd.DataFrame(result_dict, index=[0])])
                else:
                    pass
            else:
                log.warning(f'Annotation ObjectType unknown')
    return df_ann


def annotation_parsing(
    datasets: sly.Project.DatasetDict,
    save_dir: str,
):
    num_cores = multiprocessing.cpu_count()
    annotation = Parallel(n_jobs=num_cores, backend='threading')(
        delayed(parse_single_annotation)(
            dataset=dataset,
            save_dir=save_dir,
        )
        for dataset in tqdm(datasets, desc='Annotation parsing')
    )

    return annotation


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='convert_sly_to_int',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    project_sly = sly.Project(cfg.data_dir, sly.OpenMode.READ)

    # 1. Annotation parsing
    df_list = annotation_parsing(
        datasets=project_sly.datasets,
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
