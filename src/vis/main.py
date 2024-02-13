from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import torchvision
from PIL import Image

from src.data.utils import CLASS_COLOR, CLASS_ID_REVERSED
from src.models.smp.model import HistologySegmentationModel

model = HistologySegmentationModel.load_from_checkpoint(
    'models/unet_se_resnet50_0102_2042/ckpt/best.ckpt',
    arch='unet',
    encoder_name='se_resnet50',
    model_name='unet_se_resnet50_0102_2042',
    in_channels=3,
    classes=[
        'Arteriole lumen',
        'Arteriole media',
        'Arteriole adventitia',
        'Capillary lumen',
        'Capillary wall',
        'Venule lumen',
        'Venule wall',
        'Immune cells',
        'Nerve trunks',
    ],
    lr=0.0001,
    optimizer_name='Adam',
    save_img_per_epoch=False,
)
model.eval()
get_tensor = torchvision.transforms.ToTensor()


def to_tensor(
    x: np.ndarray,
) -> np.ndarray:
    return x.transpose([2, 0, 1]).astype('float32')


def processing(*args):
    (source_image,) = args

    return inference(source_image=source_image.copy())


def get_img_mask_union(
    img_0: np.ndarray,
    alpha_0: float,
    img_1: np.ndarray,
    alpha_1: float,
    color: Tuple[int, int, int],
) -> np.ndarray:
    return cv2.addWeighted(
        np.array(img_0).astype('uint8'),
        alpha_0,
        (cv2.cvtColor(np.array(img_1).astype('uint8'), cv2.COLOR_GRAY2RGB) * color).astype(
            np.uint8,
        ),
        alpha_1,
        0,
    )


def inference(
    source_image: Image.Image,
):
    source_image = np.array(source_image)
    source_image = cv2.resize(source_image, (896, 896))

    image = to_tensor(np.array(source_image.copy()))
    y_hat = model(torch.Tensor([image]).to('cuda'))
    mask_pred = y_hat.sigmoid()
    mask_pred = (mask_pred > 0.5).float()
    mask_pred = mask_pred.permute(0, 2, 3, 1)
    mask_pred = mask_pred.squeeze().cpu().numpy().round()

    color_mask_pred = np.zeros(source_image.shape)
    color_mask_pred[:, :] = (128, 128, 128)

    for class_id in CLASS_ID_REVERSED:
        m = np.zeros((source_image.shape[1], source_image.shape[0]))
        m[mask_pred[:, :, class_id - 1] == 1] = 1

        source_image = get_img_mask_union(
            img_0=source_image,
            alpha_0=1,
            img_1=m,
            alpha_1=0.5,
            color=CLASS_COLOR[CLASS_ID_REVERSED[class_id]],
        )

        color_mask_pred[mask_pred[:, :, class_id - 1] == 1] = CLASS_COLOR[
            CLASS_ID_REVERSED[class_id]
        ]

    return Image.fromarray(np.array(source_image).astype('uint8')), Image.fromarray(
        np.array(color_mask_pred).astype('uint8'),
    )


def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    source_image = gr.Image(
                        label='Source Image',
                        image_mode='RGB',
                        type='pil',
                        sources=['upload'],
                        height=640,
                    )
                with gr.Row():
                    start = gr.Button('Start')
            with gr.Column(scale=1):
                results = gr.Gallery(height=640, columns=1, object_fit='scale-down')

        start.click(
            processing,
            inputs=[
                source_image,
            ],
            outputs=results,
        )

    demo.launch(
        server_name='0.0.0.0',
        server_port=7883,
    )


if __name__ == '__main__':
    main()
