import json
import logging
import os
import time
from glob import glob

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from src import PROJECT_DIR
from src.models.smp.model import HistologySegmentationModel
from src.models.smp.predict_image_DELETE_ME import prediction_model, processing_mask
from src.models.smp.utils import preprocessing_img

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='predict',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.warning(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    data_dir = os.path.join(PROJECT_DIR, cfg.data_dir)
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)
    model_dir = os.path.join(PROJECT_DIR, cfg.model_dir)

    # TODO: check what you need and don't need from model_config
    # Load the model with its configuration and weights
    model_weights = os.path.join(model_dir, 'weights.ckpt')
    with open(os.path.join(model_dir, 'config.json'), 'r') as file:
        model_cfg = json.load(file)
        print(model_cfg)  # TODO: remove it after debug
    start = time.time()
    model = HistologySegmentationModel.load_from_checkpoint(
        checkpoint_path=model_weights,
        arch=cfg.architecture,
        encoder_name=cfg.encoder,
        model_name=cfg.model_name,
        in_channels=3,
        classes=cfg.classes,
        map_location='cuda:0' if cfg.device == 'cuda' else cfg.device,
    )
    model.eval()
    log.info(
        f'Model "{cfg.model_name}" loaded successfully. Time taken: {time.time() - start:.1f} s',
    )

    # Process images
    start_inference = time.time()
    os.makedirs(save_dir, exist_ok=True)
    images_batch, images_input, images_name = [], [], []
    images_path = glob(f'{data_dir}/*.[pj][np][ge]*')
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
                # TODO: think of this method instead -> model.predict(images)
                masks = prediction_model(
                    model=model,
                    images=np.array(images_batch),
                    device=cfg.device,
                )
                for idx, (mask, image_input) in enumerate(zip(masks, images_input)):
                    overlay, color_mask = processing_mask(
                        image_input=image_input,
                        input_size=cfg.input_size,
                        classes=cfg.classes,
                        mask=mask,
                    )
                    color_mask.save(f'{save_dir}/{images_name[idx]}_mask.png')
                    overlay.save(f'{save_dir}/{images_name[idx]}_overlay.png')
                images_batch, images_input, images_name = [], [], []
                pbar.update(cfg.batch_size)
        if len(images_input) != 0:
            masks = prediction_model(
                model=model,
                images=np.array(images_batch),
                device=cfg.device,
            )
            for idx, (mask, image_input) in enumerate(zip(masks, images_input)):
                overlay, color_mask = processing_mask(
                    image_input=image_input,
                    input_size=cfg.input_size,
                    classes=cfg.classes,
                    mask=mask,
                )
                color_mask.save(f'{save_dir}/{images_name[idx]}_mask.png')
                overlay.save(f'{save_dir}/{images_name[idx]}_overlay.png')
            pbar.update(len(images_input))
    log.info(f'Prediction time: {time.time() - start_inference:.1f} s')
    log.info(f'Overall computation time: {time.time() - start:.1f} s')


if __name__ == '__main__':
    main()
