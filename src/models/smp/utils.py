import os
from csv import DictWriter
from typing import List, Tuple

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import wandb

from src.data.utils import CLASS_COLOR, CLASS_ID, CLASS_ID_REVERSED


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
    dice = 2 * iou.cpu().numpy() / (iou.cpu().numpy() + 1)
    f1 = smp.metrics.f1_score(tp, fp, fn, tn)
    precision = smp.metrics.precision(tp, fp, fn, tn)
    sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn)
    specificity = smp.metrics.specificity(tp, fp, fn, tn)
    return {
        'loss': loss.detach().cpu().numpy(),
        'tp': tp.cpu().numpy(),
        'fp': fp.cpu().numpy(),
        'fn': fn.cpu().numpy(),
        'tn': tn.cpu().numpy(),
        'IoU': iou.cpu().numpy(),
        'Dice': dice,
        'F1': f1.cpu().numpy(),
        'Recall': sensitivity.cpu().numpy(),
        'Precision': precision.cpu().numpy(),
        'Sensitivity': sensitivity.cpu().numpy(),
        'Specificity': specificity.cpu().numpy(),
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
        f'{split}/IoU (mean)': metrics['IoU'].mean(),
        f'{split}/Dice (mean)': metrics['Dice'].mean(),
        f'{split}/Precision (mean)': metrics['Precision'].mean(),
        f'{split}/Recall (mean)': metrics['Recall'].mean(),
        f'{split}/Sensitivity (mean)': metrics['Sensitivity'].mean(),
        f'{split}/Specificity (mean)': metrics['Specificity'].mean(),
        f'{split}/F1 (mean)': metrics['Specificity'].mean(),
        f'IoU {split}/mean': metrics['IoU'].mean(),
        f'Dice {split}/mean': metrics['Dice'].mean(),
        f'Precision {split}/mean': metrics['Precision'].mean(),
        f'Recall {split}/mean': metrics['Recall'].mean(),
        f'Sensitivity {split}/mean': metrics['Sensitivity'].mean(),
        f'Specificity {split}/mean': metrics['Specificity'].mean(),
        f'F1 {split}/mean': metrics['Specificity'].mean(),
    }

    metrics_l = metrics_log.copy()
    metrics_l['epoch'] = epoch
    wandb.log(
        metrics_l,
    )

    with open(f'models/{model_name}/metrics.csv', 'a', newline='') as f_object:
        fieldnames = [
            'Epoch',
            'IoU',
            'Dice',
            'Precision',
            'Recall',
            'Sensitivity',
            'Specificity',
            'F1',
            'Split',
            'Class',
        ]
        writer = DictWriter(f_object, fieldnames=fieldnames)
        if header_w:
            writer.writeheader()

        for num, cl in enumerate(classes):
            for metric_name in [
                'IoU',
                'Dice',
                'F1',
                'Precision',
                'Recall',
                'Sensitivity',
                'Specificity',
            ]:
                metrics_log[f'{split}/{metric_name} ({cl})'] = metrics[metric_name][num]
                metrics_log[f'{metric_name} {split}/{cl}'] = metrics[metric_name][num]
            writer.writerow(
                {
                    'Epoch': epoch,
                    'IoU': metrics['IoU'][num],
                    'Dice': metrics['Dice'][num],
                    'Precision': metrics['Precision'][num],
                    'Recall': metrics['Recall'][num],
                    'Sensitivity': metrics['Sensitivity'][num],
                    'Specificity': metrics['Specificity'][num],
                    'F1': metrics['F1'][num],
                    'Split': split,
                    'Class': cl,
                },
            )
        writer.writerow(
            {
                'IoU': metrics['IoU'].mean(),
                'Dice': metrics['Dice'].mean(),
                'Precision': metrics['Precision'].mean(),
                'Recall': metrics['Recall'].mean(),
                'Sensitivity': metrics['Sensitivity'].mean(),
                'Specificity': metrics['Specificity'].mean(),
                'F1': metrics['F1'].mean(),
                'Split': split,
                'Class': 'Mean',
            },
        )
        log_dict(metrics_log, on_epoch=True)
        f_object.close()


def log_predict_model_on_epoch(
    img,
    mask,
    pred_mask,
    classes,
    epoch,
    model_name,
):
    img = img.permute(0, 2, 3, 1)
    img = img.squeeze().cpu().numpy().round()
    mask = mask.squeeze().cpu().numpy().round()
    pred_mask = pred_mask.squeeze().cpu().numpy().round()
    wandb_images = []

    for idx, (img_, mask_, pr_mask) in enumerate(zip(img, mask, pred_mask)):
        img_ = np.array(img_)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

        color_mask_gr = np.zeros(img_.shape)
        color_mask_pred = np.zeros(img_.shape)
        color_mask_pred[:, :] = (128, 128, 128)
        color_mask_gr[:, :] = (128, 128, 128)

        wandb_mask_inference = np.zeros((img_.shape[0], img_.shape[1]))
        wandb_mask_ground_truth = np.zeros((img_.shape[0], img_.shape[1]))
        for cl, m, m_p in zip(classes, mask_, pr_mask):
            color_mask_gr[m[:, :] == 1] = CLASS_COLOR[cl]
            color_mask_pred[m_p[:, :] == 1] = CLASS_COLOR[cl]
            wandb_mask_inference[m_p[:, :] == 1] = CLASS_ID[cl]
            wandb_mask_ground_truth[m[:, :] == 1] = CLASS_ID[cl]

        res = np.hstack((img_, color_mask_gr))
        res = np.hstack((res, color_mask_pred))

        cv2.imwrite(
            f'models/{model_name}/images_per_epoch/Experiment_{str(idx).zfill(2)}_epoch_{str(epoch).zfill(3)}.png',
            cv2.cvtColor(res.astype('uint8'), cv2.COLOR_RGB2BGR),
        )

        wandb_images.append(
            wandb.Image(
                img_,
                masks={
                    'predictions': {
                        'mask_data': wandb_mask_inference,
                        'class_labels': CLASS_ID_REVERSED,
                    },
                    'ground_truth': {
                        'mask_data': wandb_mask_ground_truth,
                        'class_labels': CLASS_ID_REVERSED,
                    },
                },
                caption=f'Example-{idx}',
            ),
        )
    wandb.log(
        {'Examples': wandb_images},
    )
