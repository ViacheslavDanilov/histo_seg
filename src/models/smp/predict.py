import json
import logging
import os
import time
from glob import glob

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from src import PROJECT_DIR
from src.models.smp.model import HistologySegmentationModel
from src.models.smp.utils import CLASS_COLOR, get_img_mask_union_pil, preprocessing_img

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_mask(
    source_image: Image,
    output_size: list[int, int],
    classes: list[str],
    classes_model: list[str],
    mask: np.ndarray,
) -> tuple[Image, Image]:
    source_image = source_image.resize((output_size[0], output_size[1]))
    color_mask = np.zeros((source_image.size[0], source_image.size[1], 3))
    mask = cv2.resize(mask, (output_size[0], output_size[1]))
    color_mask[:, :] = (128, 128, 128)

    for idx, cl in enumerate(classes_model):
        if cl in classes:
            source_image = get_img_mask_union_pil(
                img=source_image,
                mask=mask[:, :, idx].copy(),
                alpha=0.85,
                color=CLASS_COLOR[cl],  # type: ignore
            )
            color_mask[mask[:, :, idx] == 1] = CLASS_COLOR[cl]
    return source_image, Image.fromarray(color_mask.astype('uint8'))


def data_batch_generator(
    data: list[str],
    batch_size: int,
):
    batch = []
    for _ in range(batch_size):
        if len(data) > 0:
            img_path = data.pop()
            batch.append(img_path)
        else:
            return batch
    return batch


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='predict',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.warning(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_path = os.path.join(PROJECT_DIR, cfg.data_path)
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)
    model_dir = os.path.join(PROJECT_DIR, cfg.model_dir)

    model_weights = f'{model_dir}/weights.ckpt'
    with open(f'{model_dir}/config.json', 'r') as file:
        model_cfg = json.load(file)
    start = time.time()
    model = HistologySegmentationModel.load_from_checkpoint(
        checkpoint_path=model_weights,
        arch=model_cfg['architecture'],
        encoder_name=model_cfg['encoder'],
        model_name=model_cfg['model_name'],
        in_channels=3,
        classes=model_cfg['classes'],
        map_location='cuda:0' if cfg.device == 'cuda' else cfg.device,
    )
    model.eval()
    log.info(
        f"Model: {model_cfg['model_name']} loaded successfully. Time taken: {time.time() - start:.1f} s",
    )

    start_inference = time.time()
    os.makedirs(save_dir, exist_ok=True)
    if os.path.isfile(data_path):
        images_path = [data_path]
    else:
        images_path = glob(f'{data_path}/*.[pj][np][ge]*')
    with tqdm(total=len(images_path), desc='Images predict') as pbar:
        while True:
            images_batch = data_batch_generator(data=images_path, batch_size=cfg.batch_size)
            if not images_batch:
                break
            images, source_images, images_name = [], [], []
            for _, img_path in enumerate(images_batch):
                images.append(
                    preprocessing_img(
                        img_path=img_path,
                        input_size=model_cfg['input_size'],
                    ),
                )
                source_images.append(Image.open(img_path))
                images_name.append(os.path.basename(img_path).split('.')[0])
            masks = model.predict(
                images=np.array(images),
                device=cfg.device,
            )
            for image_name, source_image, mask in zip(images_name, source_images, masks):
                overlay, color_mask = process_mask(
                    source_image=source_image,
                    output_size=cfg.output_size,
                    classes_model=model_cfg['classes'],
                    classes=cfg.classes,
                    mask=mask,
                )
                color_mask.save(f'{save_dir}/{image_name}_mask.png')
                overlay.save(f'{save_dir}/{image_name}_overlay.png')
            pbar.update(len(images_batch))
    log.info(f'Prediction time: {time.time() - start_inference:.1f} s')
    log.info(f'Overall computation time: {time.time() - start:.1f} s')


if __name__ == '__main__':
    main()
