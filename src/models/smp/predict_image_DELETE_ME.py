import logging
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from src.data.utils import CLASS_COLOR
from src.models.smp.model import HistologySegmentationModel
from src.models.smp.utils import get_img_mask_union_pil, preprocessing_img

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def prediction_model(
    model: HistologySegmentationModel,
    images: np.ndarray,
    device: str,
):
    y_hat = model(torch.Tensor(images).to(device)).cpu().detach()
    masks = y_hat.sigmoid()
    masks = (masks > 0.5).float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.squeeze().numpy().round()
    return masks


def process_mask(
    image_input: Image,
    input_size: int,
    classes: list[str],
    mask: np.ndarray,
) -> tuple[Image, Image]:
    image_input = image_input.resize((input_size, input_size))
    color_mask = np.zeros((image_input.size[0], image_input.size[1], 3))
    color_mask[:, :] = (128, 128, 128)

    for idx, cl in enumerate(zip(classes)):
        class_name = cl[0]
        image_input = get_img_mask_union_pil(
            img=image_input,
            mask=mask[:, :, idx].copy(),
            alpha=0.85,
            color=CLASS_COLOR[class_name],  # type: ignore
        )
        color_mask[mask[:, :, idx] == 1] = CLASS_COLOR[class_name]
    return image_input, Image.fromarray(color_mask.astype('uint8'))


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='predict_image_DELETE_ME',
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
    image = preprocessing_img(
        img_path=cfg.img_path,
        input_size=cfg.input_size,
    )
    mask = prediction_model(
        model=model,
        images=np.array([image]),
        device=cfg.device,
    )
    log.info(f'Inference time: {time.time() - start_inference}s')
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    img_name = os.path.basename(cfg.img_path).split('.')[0]
    image_input = Image.open(cfg.img_path)
    union_mask, color_mask = process_mask(
        image_input=image_input,
        input_size=cfg.input_size,
        classes=cfg.classes,
        mask=mask,
    )
    color_mask.save(f'{cfg.save_dir}/{img_name}_color_mask.png')
    union_mask.save(f'{cfg.save_dir}/{img_name}_union_mask.png')
    log.info(f'Summary compute time: {time.time() - start}')


if __name__ == '__main__':
    main()
