import os
from csv import DictWriter
from typing import List, Tuple

import cv2
import numpy as np
import segmentation_models_pytorch as smp

from src.data.utils import CLASS_COLOR


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


def get_metrics(
    mask,
    pred_mask,
    loss,
    classes,
):
    tp, fp, fn, tn = smp.metrics.get_stats(
        pred_mask.long(),
        mask.long(),
        mode='multilabel',
        num_classes=len(classes),
    )
    iou = smp.metrics.iou_score(tp, fp, fn, tn)
    precision = smp.metrics.precision(tp, fp, fn, tn)
    sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn)
    specificity = smp.metrics.specificity(tp, fp, fn, tn)
    f1 = smp.metrics.f1_score(tp, fp, fn, tn)
    dice_score = 1 - loss

    return {
        'dice_score': dice_score.detach().cpu().numpy(),
        'loss': loss.detach().cpu().numpy(),
        'IOU': iou.cpu().numpy(),
        'Precision': precision.cpu().numpy(),
        'Sensitivity': sensitivity.cpu().numpy(),
        'Specificity': specificity.cpu().numpy(),
        'F1': f1.cpu().numpy(),
        'tp': tp.cpu().numpy(),
        'fp': fp.cpu().numpy(),
        'fn': fn.cpu().numpy(),
        'tn': tn.cpu().numpy(),
    }


def save_metrics_on_epoch(
    metrics_epoch: List[dict],
    split: str,
    model_name: str,
    classes: List[str],
    epoch: int,
    log_dict,
) -> None:
    header_w = False
    if not os.path.exists(f'models/{model_name}/metrics.csv'):
        header_w = True

    metrics_name = metrics_epoch[0].keys()
    metrics = {}
    for metric_name in metrics_name:
        for batch in metrics_epoch:
            if metric_name not in metrics:
                metrics[metric_name] = (
                    batch[metric_name]
                    if batch[metric_name].size == 1
                    else np.mean(
                        batch[metric_name],
                        axis=0,
                    )
                )
            else:
                if batch[metric_name].size == 1:
                    metrics[metric_name] = np.mean((batch[metric_name], metrics[metric_name]))
                else:
                    metrics[metric_name] = np.mean(
                        (np.mean(batch[metric_name], axis=0), metrics[metric_name]),
                        axis=0,
                    )

    metrics_log = {
        f'{split}/IOU (mean)': metrics['IOU'].mean(),
        f'{split}/Precision (mean)': metrics['Precision'].mean(),
        f'{split}/Sensitivity (mean)': metrics['Sensitivity'].mean(),
        f'{split}/Specificity (mean)': metrics['Specificity'].mean(),
        f'{split}/Dice score': metrics['dice_score'],
        f'IOU {split}/mean': metrics['IOU'].mean(),
        f'Precision {split}/mean': metrics['Precision'].mean(),
        f'Sensitivity {split}/mean': metrics['Sensitivity'].mean(),
        f'Specificity {split}/mean': metrics['Specificity'].mean(),
    }

    with open(f'models/{model_name}/metrics.csv', 'a', newline='') as f_object:
        fieldnames = [
            'Epoch',
            'Metric',
            'Class',
            'Split',
            'Value',
        ]
        writer = DictWriter(f_object, fieldnames=fieldnames)
        if header_w:
            writer.writeheader()

        for metric_name in ['IOU', 'Precision', 'Sensitivity', 'Specificity', 'F1']:
            for num, cl in enumerate(classes):
                metrics_log[f'{split}/{metric_name} ({cl})'] = metrics[metric_name][num]
                metrics_log[f'{metric_name} {split}/{cl}'] = metrics[metric_name][num]
                writer.writerow(
                    {
                        'Epoch': epoch + 1,
                        'Metric': metric_name,
                        'Class': cl,
                        'Split': split,
                        'Value': metrics[metric_name][num],
                    },
                )
            writer.writerow(
                {
                    'Epoch': epoch + 1,
                    'Metric': metric_name,
                    'Class': 'Mean',
                    'Split': split,
                    'Value': metrics[metric_name].mean(),
                },
            )
        log_dict(metrics_log, on_epoch=True)
        f_object.close()


def log_predict_model_on_epoch(
    img,
    mask,
    pred_mask,
    classes,
    my_logger,
    epoch,
):
    img = img.permute(0, 2, 3, 1)
    img = img.squeeze().cpu().numpy().round()
    mask = mask.squeeze().cpu().numpy().round()
    pred_mask = pred_mask.squeeze().cpu().numpy().round()
    for idy, (img_, mask_, pr_mask) in enumerate(zip(img, mask, pred_mask)):
        img_ = np.array(img_)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

        color_mask_gr = np.zeros(img_.shape)
        color_mask_pred = np.zeros(img_.shape)
        color_mask_pred[:, :] = (128, 128, 128)
        color_mask_gr[:, :] = (128, 128, 128)

        for cl, m, m_p in zip(classes, mask_, pr_mask):
            color_mask_gr[m[:, :] == 1] = CLASS_COLOR[cl]
            color_mask_pred[m_p[:, :] == 1] = CLASS_COLOR[cl]

        res = np.hstack((img_, color_mask_gr))
        res = np.hstack((res, color_mask_pred))

        # cv2.imwrite(
        #     f'data/experiment/all/Experiment_{str(idy).zfill(2)}_epoch_{str(epoch).zfill(3)}.png',
        #     cv2.cvtColor(res, cv2.COLOR_RGB2BGR),
        # )

        my_logger.report_image(
            'All class',
            f'Experiment {idy}',
            image=res,
            iteration=epoch,
        )
