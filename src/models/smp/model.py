from typing import List

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from clearml import Logger

from src.models.smp.utils import get_metrics, log_predict_model_on_epoch, save_metrics_on_epoch


class HistologySegmentationModel(pl.LightningModule):
    """The model dedicated to the segmentation of histopathological images."""

    def __init__(
        self,
        arch: str,
        encoder_name: str,
        model_name: str,
        in_channels: int,
        classes: List[str],
        lr: float,
        optimizer_name: str,
        save_img_per_epoch: bool,
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
        self.epoch = 0
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))
        self.training_step_outputs = []  # type: ignore
        self.validation_step_outputs = []  # type: ignore
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        self.model_name = model_name
        self.lr = lr
        self.optimizer = optimizer_name
        self.save_img_per_epoch = save_img_per_epoch
        self.my_logger = Logger.current_logger()

    def forward(
        self,
        image: torch.tensor,
    ) -> dict:
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def training_step(
        self,
        batch: List[float],
        batch_idx: int,
    ) -> dict:
        img, mask = batch
        logits_mask = self.forward(img)

        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()  # type: ignore
        pred_mask = (prob_mask > 0.5).float()

        self.log('training/loss', loss, prog_bar=True, on_epoch=True)
        self.training_step_outputs.append(
            get_metrics(
                mask=mask,
                pred_mask=pred_mask,
                loss=loss,
                classes=self.classes,
            ),
        )
        return {
            'loss': loss,
        }

    def on_train_epoch_end(self):
        save_metrics_on_epoch(
            metrics_epoch=self.training_step_outputs,
            split='train',
            model_name=self.model_name,
            classes=self.classes,
            epoch=self.epoch,
            log_dict=self.log_dict,
        )
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
        if batch_idx == 0 and self.save_img_per_epoch:
            log_predict_model_on_epoch(
                img=img,
                mask=mask,
                pred_mask=pred_mask,
                classes=self.classes,
                my_logger=self.my_logger,
                epoch=self.epoch,
                model_name=self.model_name,
            )

    def on_validation_epoch_end(self):
        save_metrics_on_epoch(
            metrics_epoch=self.validation_step_outputs,
            split='test',
            model_name=self.model_name,
            classes=self.classes,
            epoch=self.epoch,
            log_dict=self.log_dict,
        )
        self.validation_step_outputs.clear()
        self.epoch += 1

    def configure_optimizers(self):
        match self.optimizer:
            case 'SGD':
                return torch.optim.SGD(self.parameters(), lr=self.lr)
            case 'RAdam':
                return torch.optim.RAdam(self.parameters(), lr=self.lr)
            case 'SAdam':
                return torch.optim.SparseAdam(self.parameters(), lr=self.lr)
            case _:
                return torch.optim.Adam(self.parameters(), lr=self.lr)
