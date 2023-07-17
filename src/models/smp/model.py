import os
from csv import DictWriter
from typing import Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from clearml import Logger


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


def log_metrics():
    pass


class OCTSegmentationModel(pl.LightningModule):
    """The model dedicated to the segmentation of OCT images."""

    # TODO: input and output types?
    def __init__(
        self,
        arch,
        encoder_name,
        in_channels,
        classes,
        colors,
        **kwargs,
    ):
        super().__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=len(classes),
            **kwargs,
        )

        self.classes = classes
        # TODO: Зачем нам цвета при обучении? Почему мы не можем просто использовать индексы классов?
        self.colors = colors
        self.epoch = 0
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))

        self.training_histogram = np.zeros(len(self.classes))
        # self.training_histogram_best_mean = np.zeros(len(self.classes))
        self.validation_histogram = np.zeros(len(self.classes))
        # self.validation_histogram_metrics = np.zeros(5)
        # self.validation_histogram_metrics_best = np.zeros(5)
        # self.validation_histogram_best_mean = np.zeros(len(self.classes))
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)

        self.my_logger = Logger.current_logger()

    # TODO: input and output types?
    def forward(
        self,
        image: torch.tensor,
    ):
        # normalize image here
        # TODO: Should you move the normalization to the OCTDataModule?
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    # TODO: input and output types?
    def training_step(
        self,
        batch,
        batch_idx,
    ):
        # if batch_idx == 0:
        #     if self.epoch > 0:
        #         self.my_logger.report_histogram(
        #             'Last IOU',
        #             'Training',
        #             iteration=self.epoch,
        #             values=self.training_histogram,
        #             xlabels=self.classes,
        #             xaxis='Classes',
        #             yaxis='IOU',
        #         )
        #         if np.mean(self.training_histogram) > np.mean(self.training_histogram_best_mean):
        #             self.training_histogram_best_mean = self.training_histogram
        #             self.my_logger.report_histogram(
        #                 'Best IOU',
        #                 'Training',
        #                 iteration=self.epoch,
        #                 values=self.training_histogram_best_mean,
        #                 xlabels=self.classes,
        #                 xaxis='Classes',
        #                 yaxis='IOU',
        #             )
        #         self.training_histogram = np.zeros(len(self.classes))

        img, mask = batch
        logits_mask = self.forward(img)

        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # tp, fp, fn, tn = smp.metrics.get_stats(
        #     pred_mask.long(),
        #     mask.long(),
        #     mode='multilabel',
        #     num_classes=len(self.classes),
        # )
        # # TODO: please add other metrics like dice, accuracy, sensitivity and specificity
        # # TODO: additional method to compute these metrics for both train and test subsets is also needed
        # iou = smp.metrics.iou_score(tp, fp, fn, tn)
        # tp = tp.cpu().numpy()
        # fp = fp.cpu().numpy()
        # fn = fn.cpu().numpy()
        # tn = tn.cpu().numpy()
        #
        # accuracy, precision, recall = np.zeros(len(self.classes)), np.zeros(len(self.classes)), np.zeros(len(self.classes))
        # for id_, cl in enumerate(self.classes):
        #     accuracy[id_] = (np.mean(tp[:, id_]) + np.mean(tn[:, id_])) / (np.mean(tp[:, id_]) + np.mean(fp[:, id_]) + np.mean(tn[:, id_]) + np.mean(fn[:, id_]))
        #     precision[id_] = np.mean(tp[:, id_]) / (np.mean(tp[:, id_]) + np.mean(fp[:, id_]))
        #     recall[id_] = np.mean(tp[:, id_]) / (np.mean(tp[:, id_]) + np.mean(fn[:, id_]))
        # dice_score = 1 - loss

        self.log('training/loss', loss, prog_bar=True, on_epoch=True)
        self.training_step_outputs.append(
            get_metrics(
                mask=mask,
                pred_mask=pred_mask,
                loss=loss,
                classes=self.classes,
            ),
        )

        # for num, cl in enumerate(self.classes):
        #     self.training_histogram[num] += iou[:, num].mean().cpu().numpy()
        #     if batch_idx != 0:
        #         self.training_histogram[num] /= 2
        #
        # metrics = {
        #     'train/IOU (mean)': iou.mean(),
        #     'train/accuracy (mean)': accuracy.mean(),
        #     'train/precision (mean)': precision.mean(),
        #     'train/recall (mean)': recall.mean(),
        #     'train/dice_score (mean)': dice_score,
        # }
        # for num, cl in enumerate(self.classes):
        #     metrics[f'train/IOU ({cl})'] = iou[:, num].mean()
        # self.log_dict(metrics, on_epoch=True)
        # self.training_step_outputs.append(get_metrics(
        #     mask=mask,
        #     pred_mask=pred_mask,
        #     loss=loss,
        #     classes=self.classes,
        # ))

        return {
            'loss': loss,
        }

    def on_train_epoch_end(self):
        metrics_name = self.training_step_outputs[0].keys()
        metrics = {}
        for metric_name in metrics_name:
            for batch in self.training_step_outputs:
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
            'train/IOU (mean)': metrics['iou'].mean(),
            'train/Precision (mean)': metrics['precision'].mean(),
            'train/Sensitivity (mean)': metrics['sensitivity'].mean(),
            'train/Specificity (mean)': metrics['specificity'].mean(),
            'train/Dice_score (mean)': metrics['dice_score'],
        }
        for num, cl in enumerate(self.classes):
            metrics_log[f'train/IOU ({cl})'] = metrics['iou'][num]
            metrics_log[f'train/Precision ({cl})'] = metrics['precision'][num]
            metrics_log[f'train/Sensitivity ({cl})'] = metrics['sensitivity'][num]
            metrics_log[f'train/Specificity ({cl})'] = metrics['specificity'][num]

            header_w = False
            if not os.path.exists(f'data/experiment/train_{cl}.csv'):
                header_w = True
            with open(f'data/experiment/train_{cl}.csv', 'a', newline='') as f_object:
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
                        'epoch': self.epoch,
                        'IOU': metrics['iou'][num],
                        'Precision': metrics['precision'][num],
                        'Sensitivity': metrics['sensitivity'][num],
                        'Specificity': metrics['specificity'][num],
                    },
                )
                f_object.close()

        self.log_dict(metrics_log, on_epoch=True)
        self.my_logger.report_histogram(
            'Last IOU',
            'Training',
            iteration=self.epoch,
            values=metrics['iou'],
            xlabels=self.classes,
            xaxis='Classes',
            yaxis='IOU',
        )
        if np.mean(self.training_histogram) < np.mean(metrics['iou']):
            self.training_histogram = metrics['iou']
            self.my_logger.report_histogram(
                'Best IOU',
                'Training',
                iteration=self.epoch,
                values=self.training_histogram,
                xlabels=self.classes,
                xaxis='Classes',
                yaxis='IOU',
            )

        header_w = False
        if not os.path.exists('data/experiment/train_mean.csv'):
            header_w = True
        with open('data/experiment/train_mean.csv', 'a', newline='') as f_object:
            fieldnames = [
                'epoch',
                'IOU (mean)',
                'Precision (mean)',
                'Sensitivity (mean)',
                'Specificity (mean)',
                'Dice_score (mean)',
            ]
            writer = DictWriter(f_object, fieldnames=fieldnames)
            if header_w:
                writer.writeheader()
            writer.writerow(
                {
                    'epoch': self.epoch,
                    'IOU (mean)': metrics['iou'].mean(),
                    'Precision (mean)': metrics['precision'].mean(),
                    'Sensitivity (mean)': metrics['sensitivity'].mean(),
                    'Specificity (mean)': metrics['specificity'].mean(),
                    'Dice_score (mean)': metrics['dice_score'],
                },
            )
            f_object.close()

        self.training_step_outputs.clear()

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        img, mask = batch
        logits_mask = self.forward(img)

        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.append(
            get_metrics(
                mask=mask,
                pred_mask=pred_mask,
                loss=loss,
                classes=self.classes,
            ),
        )

        if batch_idx == 0:
            img = img.permute(0, 2, 3, 1)
            img = img.squeeze().cpu().numpy().round()
            mask = mask.squeeze().cpu().numpy().round()
            pred_mask = pred_mask.squeeze().cpu().numpy().round()
            for idy, (img_, mask_, pr_mask) in enumerate(zip(img, mask, pred_mask)):
                img_ = np.array(img_)
                img_g = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                img_p = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                img_0 = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                for cl, m, m_p in zip(self.classes, mask_, pr_mask):
                    # Groundtruth
                    img_g = get_img_mask_union(
                        img_0=img_g,
                        alpha_0=1,
                        img_1=m,
                        alpha_1=0.5,
                        color=self.colors[cl],
                    )
                    img_g_cl = get_img_mask_union(
                        img_0=img_0.copy(),
                        alpha_0=1,
                        img_1=m,
                        alpha_1=0.5,
                        color=self.colors[cl],
                    )
                    # Prediction
                    img_p = get_img_mask_union(
                        img_0=img_p,
                        alpha_0=1,
                        img_1=m_p,
                        alpha_1=0.5,
                        color=self.colors[cl],
                    )
                    img_p_cl = get_img_mask_union(
                        img_0=img_0.copy(),
                        alpha_0=1,
                        img_1=m_p,
                        alpha_1=0.5,
                        color=self.colors[cl],
                    )
                    res = np.hstack((img_0, img_g_cl))
                    res = np.hstack((res, img_p_cl))
                    self.my_logger.report_image(
                        cl,
                        f'Experiment {idy}',
                        image=res,
                        iteration=self.epoch,
                    )
                res = np.hstack((img_0, img_g))
                res = np.hstack((res, img_p))

                cv2.imwrite(
                    f'data/experiment/all/Experiment_{str(idy).zfill(2)}_epoch_{str(self.epoch).zfill(3)}.png',
                    cv2.cvtColor(res, cv2.COLOR_RGB2BGR),
                )

                self.my_logger.report_image(
                    'All class',
                    f'Experiment {idy}',
                    image=res,
                    iteration=self.epoch,
                )

        # tp, fp, fn, tn = smp.metrics.get_stats(
        #     pred_mask.long(),
        #     mask.long(),
        #     mode='multilabel',
        #     num_classes=len(self.classes),
        # )
        # iou = smp.metrics.iou_score(tp, fp, fn, tn)
        # tp = tp.cpu().numpy()
        # fp = fp.cpu().numpy()
        # fn = fn.cpu().numpy()
        # tn = tn.cpu().numpy()
        #
        # accuracy, precision, recall = np.zeros(len(self.classes)), np.zeros(len(self.classes)), np.zeros(len(self.classes))
        # for id_, cl in enumerate(self.classes):
        #     accuracy[id_] = (np.mean(tp[:, id_]) + np.mean(tn[:, id_])) / (np.mean(tp[:, id_]) + np.mean(fp[:, id_]) + np.mean(tn[:, id_]) + np.mean(fn[:, id_]))
        #     precision[id_] = np.mean(tp[:, id_]) / (np.mean(tp[:, id_]) + np.mean(fp[:, id_]))
        #     recall[id_] = np.mean(tp[:, id_]) / (np.mean(tp[:, id_]) + np.mean(fn[:, id_]))
        # dice_score = 1 - loss
        #
        # self.validation_histogram_metrics[0] = accuracy.mean()
        # if self.validation_histogram_metrics_best[0] < accuracy.mean():
        #     self.validation_histogram_metrics_best[0] = accuracy.mean()
        # self.validation_histogram_metrics[1] = precision.mean()
        # if self.validation_histogram_metrics_best[1] < precision.mean():
        #     self.validation_histogram_metrics_best[1] = precision.mean()
        # self.validation_histogram_metrics[2] = recall.mean()
        # if self.validation_histogram_metrics_best[2] < recall.mean():
        #     self.validation_histogram_metrics_best[2] = recall.mean()
        # self.validation_histogram_metrics[3] = iou.mean()
        # if self.validation_histogram_metrics_best[3] < iou.mean():
        #     self.validation_histogram_metrics_best[3] = iou.mean()
        # self.validation_histogram_metrics[4] = dice_score
        # if self.validation_histogram_metrics_best[4] < dice_score:
        #     self.validation_histogram_metrics_best[4] = dice_score
        #
        # metrics = {
        #     'test/IOU (mean)': iou.mean(),
        #     'test/accuracy (mean)': accuracy.mean(),
        #     'test/precision (mean)': precision.mean(),
        #     'test/recall (mean)': recall.mean(),
        #     'test/dice_score (mean)': dice_score,
        # }
        # for num, cl in enumerate(self.classes):
        #     metrics[f'test/IOU ({cl})'] = iou[:, num].mean()
        # self.log_dict(metrics, on_epoch=True)
        # self.log('test/loss', loss, prog_bar=True, on_epoch=True)
        #
        # if batch_idx == 0:
        #     if self.epoch > 0:
        #         self.my_logger.report_histogram(
        #             'Last Metrics',
        #             'Test',
        #             iteration=self.epoch,
        #             values=self.validation_histogram_metrics,
        #             xlabels=['Accuracy', 'Precision', 'Recall', 'IOU', 'Dice_score'],
        #             xaxis='Metrics',
        #             yaxis='variable',
        #         )
        #         self.my_logger.report_histogram(
        #             'Best Metrics',
        #             'Test',
        #             values=self.validation_histogram_metrics_best,
        #             xlabels=['Accuracy', 'Precision', 'Recall', 'IOU', 'Dice_score'],
        #             xaxis='Metrics',
        #             yaxis='variable',
        #         )
        #         if np.mean(self.validation_histogram) > np.mean(
        #             self.validation_histogram_best_mean,
        #         ):
        #             self.validation_histogram_best_mean = self.validation_histogram
        #             self.my_logger.report_histogram(
        #                 'Best IOU',
        #                 'Test',
        #                 iteration=self.epoch,
        #                 values=self.validation_histogram_best_mean,
        #                 xlabels=self.classes,
        #                 xaxis='Classes',
        #                 yaxis='IOU',
        #             )
        #         self.validation_histogram = np.zeros(len(self.classes))
        #
        #     self.epoch += 1
        #
        #     img = img.permute(0, 2, 3, 1)
        #     img = img.squeeze().cpu().numpy().round()
        #     mask = mask.squeeze().cpu().numpy().round()
        #     pred_mask = pred_mask.squeeze().cpu().numpy().round()
        #     for idy, (img_, mask_, pr_mask) in enumerate(zip(img, mask, pred_mask)):
        #         img_ = np.array(img_)
        #         img_g = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        #         img_p = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        #         img_0 = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        #         for cl, m, m_p in zip(self.classes, mask_, pr_mask):
        #             # Groundtruth
        #             img_g = get_img_mask_union(
        #                 img_0=img_g,
        #                 alpha_0=1,
        #                 img_1=m,
        #                 alpha_1=0.5,
        #                 color=self.colors[cl],
        #             )
        #             img_g_cl = get_img_mask_union(
        #                 img_0=img_0.copy(),
        #                 alpha_0=1,
        #                 img_1=m,
        #                 alpha_1=0.5,
        #                 color=self.colors[cl],
        #             )
        #             # Prediction
        #             img_p = get_img_mask_union(
        #                 img_0=img_p,
        #                 alpha_0=1,
        #                 img_1=m_p,
        #                 alpha_1=0.5,
        #                 color=self.colors[cl],
        #             )
        #             img_p_cl = get_img_mask_union(
        #                 img_0=img_0.copy(),
        #                 alpha_0=1,
        #                 img_1=m_p,
        #                 alpha_1=0.5,
        #                 color=self.colors[cl],
        #             )
        #             res = np.hstack((img_0, img_g_cl))
        #             res = np.hstack((res, img_p_cl))
        #             self.my_logger.report_image(
        #                 cl,
        #                 f'Experiment {idy}',
        #                 image=res,
        #                 iteration=self.epoch,
        #             )
        #         res = np.hstack((img_0, img_g))
        #         res = np.hstack((res, img_p))
        #         self.my_logger.report_image(
        #             'All class',
        #             f'Experiment {idy}',
        #             image=res,
        #             iteration=self.epoch,
        #         )
        #
        # for num, cl in enumerate(self.classes):
        #     self.validation_histogram[num] += iou[:, num].mean().cpu().numpy()
        #     if batch_idx != 0:
        #         self.validation_histogram[num] /= 2

    def on_validation_epoch_end(self):
        metrics_name = self.validation_step_outputs[0].keys()
        metrics = {}
        for metric_name in metrics_name:
            for batch in self.validation_step_outputs:
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
            'test/IOU (mean)': metrics['iou'].mean(),
            'test/Precision (mean)': metrics['precision'].mean(),
            'test/Sensitivity (mean)': metrics['sensitivity'].mean(),
            'test/Specificity (mean)': metrics['specificity'].mean(),
            'test/Dice_score (mean)': metrics['dice_score'],
        }
        for num, cl in enumerate(self.classes):
            metrics_log[f'test/IOU ({cl})'] = metrics['iou'][num]
            metrics_log[f'test/Precision ({cl})'] = metrics['precision'][num]
            metrics_log[f'test/Sensitivity ({cl})'] = metrics['sensitivity'][num]
            metrics_log[f'test/Specificity ({cl})'] = metrics['specificity'][num]

            header_w = False
            if not os.path.exists(f'data/experiment/val_{cl}.csv'):
                header_w = True
            with open(f'data/experiment/val_{cl}.csv', 'a', newline='') as f_object:
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
                        'epoch': self.epoch,
                        'IOU': metrics['iou'][num],
                        'Precision': metrics['precision'][num],
                        'Sensitivity': metrics['sensitivity'][num],
                        'Specificity': metrics['specificity'][num],
                    },
                )
                f_object.close()

        self.log_dict(metrics_log, on_epoch=True)
        self.my_logger.report_histogram(
            'Last Metrics',
            'Test',
            iteration=self.epoch,
            values=[
                metrics['precision'].mean(),
                metrics['sensitivity'].mean(),
                metrics['specificity'].mean(),
                metrics['dice_score'].mean(),
            ],
            xlabels=['Precision', 'Sensitivity', 'Specificity', 'Dice_score'],
            xaxis='Metrics',
            yaxis='variable',
        )
        if np.mean(self.validation_histogram) > np.mean(metrics['iou']):
            self.validation_histogram = metrics['iou']
            self.my_logger.report_histogram(
                'Best IOU',
                'Test',
                iteration=self.epoch,
                values=self.validation_histogram,
                xlabels=self.classes,
                xaxis='Classes',
                yaxis='IOU',
            )

        header_w = False
        if not os.path.exists('data/experiment/val_mean.csv'):
            header_w = True
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
            if header_w:
                writer.writeheader()
            writer.writerow(
                {
                    'epoch': self.epoch,
                    'IOU (mean)': metrics['iou'].mean(),
                    'Precision (mean)': metrics['precision'].mean(),
                    'Sensitivity (mean)': metrics['sensitivity'].mean(),
                    'Specificity (mean)': metrics['specificity'].mean(),
                    'Dice_score (mean)': metrics['dice_score'],
                },
            )
            f_object.close()

        self.validation_step_outputs.clear()
        self.epoch += 1

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00012)
