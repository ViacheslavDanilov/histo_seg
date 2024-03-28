import logging
import os
import time
from glob import glob

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from src.models.smp.inference_img import prediction_model, processing_mask
from src.models.smp.model import HistologySegmentationModel
from src.models.smp.utils import preprocessing_img

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='inference_batch',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.warning(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    start = time.time()
    model = HistologySegmentationModel.load_from_checkpoint(
        cfg.model_weights,
        arch=cfg.architecture,
        encoder_name=cfg.encoder,
        model_name=cfg.model_name,
        in_channels=3,
        classes=cfg.classes,
        map_location='cuda:0' if cfg.device == 'cuda' else cfg.device,
    )
    model.eval()
    log.info(f'Model {cfg.model_name} loaded success, time: {time.time() - start}s')
    start_inference = time.time()
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    images_batch, images_input, images_name = [], [], []
    images_path = glob(f'{cfg.data_dir}/*.[pj][np][ge]*')
    with tqdm(total=len(images_path), desc='Images predict') as pbar:
        for _, img_path in enumerate(images_path):
            images_batch.append(
                preprocessing_img(
                    img_path=img_path,
                    input_size=cfg.input_size,
                ),
            )
            image_input = Image.open(img_path)
            images_input.append(
                image_input.resize((cfg.input_size, cfg.input_size)),
            )
            images_name.append(os.path.basename(img_path).split('.')[0])
            if len(images_batch) == cfg.batch_size:
                masks = prediction_model(
                    model=model,
                    images=np.array(images_batch),
                    device=cfg.device,
                )
                for idx, (mask, image_input) in enumerate(zip(masks, images_input)):
                    union_mask, color_mask = processing_mask(
                        image_input=image_input,
                        input_size=cfg.input_size,
                        classes=cfg.classes,
                        mask=mask,
                    )
                    color_mask.save(f'{cfg.save_dir}/{images_name[idx]}_color_mask.png')
                    union_mask.save(f'{cfg.save_dir}/{images_name[idx]}_union_mask.png')
                images_batch, images_input, images_name = [], [], []
                pbar.update(cfg.batch_size)
        if len(images_input) != 0:
            masks = prediction_model(
                model=model,
                images=np.array(images_batch),
                device=cfg.device,
            )
            for idx, (mask, image_input) in enumerate(zip(masks, images_input)):
                union_mask, color_mask = processing_mask(
                    image_input=image_input,
                    input_size=cfg.input_size,
                    classes=cfg.classes,
                    mask=mask,
                )
                color_mask.save(f'{cfg.save_dir}/{images_name[idx]}_color_mask.png')
                union_mask.save(f'{cfg.save_dir}/{images_name[idx]}_union_mask.png')
            pbar.update(len(images_input))
    log.info(f'Inference time: {time.time() - start_inference}s')
    log.info(f'Summary compute time: {time.time() - start}')


if __name__ == '__main__':
    main()
