import json
import logging
import multiprocessing
import os
import shutil
from typing import Any, List, Tuple

import hydra
import numpy as np
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from supervisely import Polygon
from tqdm import tqdm

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_mask_properties(
    figure: dict,
    mask: np.ndarray,
) -> Tuple[str, Polygon, List[List[Any]]]:
    if figure['geometryType'] == 'bitmap':
        mask = mask.astype(bool)
        bitmap = figure['bitmap']['data']
        mask_ = sly.Bitmap.base64_2_data(bitmap)
        mask[
            figure['bitmap']['origin'][1] : figure['bitmap']['origin'][1] + mask_.shape[0],
            figure['bitmap']['origin'][0] : figure['bitmap']['origin'][0] + mask_.shape[1],
        ] = mask_[:, :]
    else:
        return None, None, None

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
    class_ids: dict,
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

        if len(ann['objects']) > 0:
            for obj in ann['objects']:
                if obj['geometryType'] == 'bitmap':
                    mask = np.zeros((ann['size']['width'], ann['size']['height']))
                    encoded_mask, contour, bbox = get_mask_properties(
                        figure=obj,
                        mask=mask,
                    )
                    if encoded_mask is not None:
                        result_dict = {
                            'image_path': f'{save_dir}/{study}/{img_name}',
                            'image_name': img_name,
                            'study': study,
                            'image_width': ann['size']['width'],
                            'image_height': ann['size']['height'],
                            'class_id': class_ids[obj['classTitle']],
                            'class_name': obj['classTitle'],
                            'x1': bbox[0][0],
                            'y1': bbox[0][1],
                            'x2': bbox[1][0],
                            'y2': bbox[1][1],
                            'xc': int(np.mean([bbox[0][0], bbox[1][0]])),
                            'yc': int(np.mean([bbox[0][1], bbox[1][1]])),
                            'box_width': bbox[1][0] - bbox[0][0],
                            'box_height': bbox[1][1] - bbox[0][1],
                            'area': int(contour.area),
                            'mask_b64': encoded_mask,
                        }
                        df_ann = pd.concat([df_ann, pd.DataFrame(result_dict, index=[0])])
                    else:
                        pass
                else:
                    pass
        else:
            pass

    return df_ann


def annotation_parsing(
    datasets: sly.Project.DatasetDict,
    class_ids: dict,
    save_dir: str,
):
    num_cores = multiprocessing.cpu_count()
    annotation = Parallel(n_jobs=num_cores, backend='threading')(
        delayed(parse_single_annotation)(
            dataset=dataset,
            class_ids=class_ids,
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
    meta = json.load(open(os.path.join(cfg.data_dir, 'meta.json')))
    # TODO: use CLASS_ID from utils.py
    class_ids = {value['title']: id for (id, value) in enumerate(meta['classes'])}

    # 1. Annotation parsing
    df_list = annotation_parsing(
        datasets=project_sly.datasets,
        class_ids=class_ids,
        save_dir=cfg.save_dir,
    )

    # 3. Save annotation data frame
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
