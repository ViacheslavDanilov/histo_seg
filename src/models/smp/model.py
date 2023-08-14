from typing import List

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from clearml import Logger

from src.models.smp.utils import get_metrics, log_predict_model_on_epoch, save_metrics_on_epoch


class HistologySegmentationModel(pl.LightningModule):
    """The model dedicated to the segmentation of OCT images."""

    # TODO: input and output types?
    def __init__(
        self,
        arch,
        encoder_name,
        in_channels,
        classes,
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
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)

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
            name='train',
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
        if batch_idx == 0:
            log_predict_model_on_epoch(
                img=img,
                mask=mask,
                pred_mask=pred_mask,
                classes=self.classes,
                my_logger=self.my_logger,
                epoch=self.epoch,
            )

    def on_validation_epoch_end(self):
        save_metrics_on_epoch(
            metrics_epoch=self.validation_step_outputs,
            name='test',
            classes=self.classes,
            epoch=self.epoch,
            log_dict=self.log_dict,
        )
        self.validation_step_outputs.clear()
        self.epoch += 1

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00012)
