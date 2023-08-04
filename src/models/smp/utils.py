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
    dice_score = 1 - loss

    return {
        'dice_score': dice_score.detach().cpu().numpy(),
        'loss': loss.detach().cpu().numpy(),
        'iou': iou.cpu().numpy(),
        'precision': precision.cpu().numpy(),
        'sensitivity': sensitivity.cpu().numpy(),
        'specificity': specificity.cpu().numpy(),
        'tp': tp.cpu().numpy(),
        'fp': fp.cpu().numpy(),
        'fn': fn.cpu().numpy(),
        'tn': tn.cpu().numpy(),
    }


def save_metrics_on_epoch(
    metrics_epoch: List[dict],
    name: str,
    classes: List[str],
    epoch: int,
    log_dict,
) -> None:
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
        f'{name}/IOU (mean)': metrics['iou'].mean(),
        f'{name}/Precision (mean)': metrics['precision'].mean(),
        f'{name}/Sensitivity (mean)': metrics['sensitivity'].mean(),
        f'{name}/Specificity (mean)': metrics['specificity'].mean(),
        f'{name}/Dice score': metrics['dice_score'],
        f'IOU {name}/mean': metrics['iou'].mean(),
        f'Precision {name}/mean': metrics['precision'].mean(),
        f'Sensitivity {name}/mean': metrics['sensitivity'].mean(),
        f'Specificity {name}/mean': metrics['specificity'].mean(),
    }
    for num, cl in enumerate(classes):
        metrics_log[f'{name}/IOU ({cl})'] = metrics['iou'][num]
        metrics_log[f'{name}/Precision ({cl})'] = metrics['precision'][num]
        metrics_log[f'{name}/Sensitivity ({cl})'] = metrics['sensitivity'][num]
        metrics_log[f'{name}/Specificity ({cl})'] = metrics['specificity'][num]
        metrics_log[f'IOU {name}/{cl}'] = metrics['iou'][num]
        metrics_log[f'Precision {name}/{cl}'] = metrics['precision'][num]
        metrics_log[f'Sensitivity {name}/{cl}'] = metrics['sensitivity'][num]
        metrics_log[f'Specificity {name}/{cl}'] = metrics['specificity'][num]

        header_w = False
        if not os.path.exists(f'data/experiment/{name}_{cl}.csv'):
            header_w = True
        with open(f'data/experiment/{name}_{cl}.csv', 'a', newline='') as f_object:
            fieldnames = [
                'epoch',
                'IOU',
                'Precision',
                'Sensitivity',
                'Specificity',
            ]
            writer = DictWriter(f_object, fieldnames=fieldnames)
            if header_w:
                writer.writeheader()
            writer.writerow(
                {
                    'epoch': epoch,
                    'IOU': metrics['iou'][num],
                    'Precision': metrics['precision'][num],
                    'Sensitivity': metrics['sensitivity'][num],
                    'Specificity': metrics['specificity'][num],
                },
            )
            f_object.close()
    log_dict(metrics_log, on_epoch=True)

    if not os.path.exists('data/experiment/val_mean.csv'):
        writer.writeheader()
    with open('data/experiment/val_mean.csv', 'a', newline='') as f_object:
        fieldnames = [
            'epoch',
            'IOU (mean)',
            'Precision (mean)',
            'Sensitivity (mean)',
            'Specificity (mean)',
            'Dice_score (mean)',
        ]
        writer = DictWriter(f_object, fieldnames=fieldnames)
        writer.writerow(
            {
                'epoch': epoch,
                'IOU (mean)': metrics['iou'].mean(),
                'Precision (mean)': metrics['precision'].mean(),
                'Sensitivity (mean)': metrics['sensitivity'].mean(),
                'Specificity (mean)': metrics['specificity'].mean(),
                'Dice_score (mean)': metrics['dice_score'],
            },
        )
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
        img_g = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_p = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_0 = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

        color_mask_gr = np.zeros(img_.shape)
        color_mask_pred = np.zeros(img_.shape)
        color_mask_pred[:, :] = (128, 128, 128)
        color_mask_gr[:, :] = (128, 128, 128)

        for cl, m, m_p in zip(classes, mask_, pr_mask):
            # Groundtruth
            img_g = get_img_mask_union(
                img_0=img_g,
                alpha_0=1,
                img_1=m,
                alpha_1=0.5,
                color=CLASS_COLOR[cl],
            )
            img_g_cl = get_img_mask_union(
                img_0=img_0.copy(),
                alpha_0=1,
                img_1=m,
                alpha_1=0.5,
                color=CLASS_COLOR[cl],
            )
            color_mask_gr[m[:, :] == 1] = CLASS_COLOR[cl]

            # Prediction
            img_p = get_img_mask_union(
                img_0=img_p,
                alpha_0=1,
                img_1=m_p,
                alpha_1=0.5,
                color=CLASS_COLOR[cl],
            )
            img_p_cl = get_img_mask_union(
                img_0=img_0.copy(),
                alpha_0=1,
                img_1=m_p,
                alpha_1=0.5,
                color=CLASS_COLOR[cl],
            )
            color_mask_pred[m_p[:, :] == 1] = CLASS_COLOR[cl]

            res = np.hstack((img_0, img_g_cl))
            res = np.hstack((res, img_p_cl))

            my_logger.report_image(
                cl,
                f'Experiment {idy}',
                image=res,
                iteration=epoch,
            )

        res = np.hstack((img_0, img_g))
        res = np.hstack((res, img_p))

        new_res = np.hstack((img_0, color_mask_gr))
        new_res = np.hstack((new_res, color_mask_pred))

        cv2.imwrite(
            f'data/experiment/all/Experiment_{str(idy).zfill(2)}_epoch_{str(epoch).zfill(3)}.png',
            cv2.cvtColor(res, cv2.COLOR_RGB2BGR),
        )

        my_logger.report_image(
            'All class',
            f'Experiment {idy}',
            image=np.vstack((res, new_res)),
            iteration=epoch,
        )
