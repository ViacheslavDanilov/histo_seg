import datetime
import logging
import os
import ssl

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.models.smp.dataset import HistologyDataModule
from src.models.smp.model import HistologySegmentationModel

ssl._create_default_https_context = ssl._create_unverified_context

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='train',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    today = datetime.datetime.today()
    task_name = f'{cfg.architecture}_{cfg.encoder}_{today.strftime("%d%m_%H%M")}'
    model_dir = os.path.join('models', f'{task_name}')

    hyperparameters = {
        'architecture': cfg.architecture,
        'encoder': cfg.encoder,
        'input_size': cfg.input_size,
        'classes': list(cfg.classes),
        'num_classes': len(cfg.classes),
        'batch_size': cfg.batch_size,
        'optimizer': cfg.optimizer,
        'lr': cfg.lr,
        'epochs': cfg.epochs,
        'device': cfg.device,
        'data_dir': cfg.data_dir,
    }

    wandb.init(
        config=hyperparameters,
        project='histology_segmentation',
    )

    callbacks = [
        LearningRateMonitor(
            logging_interval='epoch',
            log_momentum=False,
        ),
    ]
    if cfg.log_artifacts:
        os.makedirs(f'{model_dir}/images_per_epoch', exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                save_top_k=5,
                monitor='val/loss',
                mode='min',
                dirpath=f'{model_dir}/ckpt/',
                filename='models_{epoch:02d}',
            ),
        )
    else:
        os.makedirs(f'{model_dir}', exist_ok=True)

    histo_data_module = HistologyDataModule(
        input_size=hyperparameters['input_size'],
        classes=cfg.classes,
        batch_size=hyperparameters['batch_size'],
        num_workers=os.cpu_count(),
        data_dir=cfg.data_dir,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir='logs/',
    )

    # Initialize model
    model = HistologySegmentationModel(
        arch=hyperparameters['architecture'],
        encoder_name=hyperparameters['encoder'],
        optimizer_name=hyperparameters['optimizer'],
        in_channels=3,
        classes=cfg.classes,
        model_name=task_name,
        lr=hyperparameters['lr'],
        save_img_per_epoch=cfg.log_artifacts,
    )

    # Initialize and run trainer
    trainer = pl.Trainer(
        devices=cfg.cuda_num,
        accelerator=cfg.device,
        max_epochs=hyperparameters['epochs'],
        logger=tb_logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=hyperparameters['batch_size'],
        default_root_dir=model_dir,
    )
    trainer.fit(
        model,
        datamodule=histo_data_module,
    )


if __name__ == '__main__':
    main()
