import datetime
import logging
import os

import hydra
import pytorch_lightning as pl
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.models.smp.dataset import HistologyDataModule
from src.models.smp.model import HistologySegmentationModel

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
    model_dir = f'models/{cfg.project_name}/{cfg.task_name}_#{today.strftime("%m-%d-%H.%M")}'

    # Initialize ClearML task
    Task.init(
        project_name=cfg.project_name,
        task_name=cfg.task_name,
        auto_connect_frameworks={'tensorboard': True, 'pytorch': True},
    )

    # Initialize data module
    oct_data_module = HistologyDataModule(
        dataset_name=cfg.dataset_name,
        project_name=cfg.project_name,
        input_size=cfg.input_size,
        classes=cfg.classes,
        batch_size=cfg.batch_size,
        num_workers=os.cpu_count(),
        data_location=cfg.data_location,
    )

    # Initialize callbacks
    checkpoint = ModelCheckpoint(
        save_top_k=5,
        monitor='val/loss',
        mode='min',
        dirpath=f'{model_dir}/ckpt/',
        filename='models_{epoch:02d}',
    )
    lr_monitor = LearningRateMonitor(
        logging_interval='epoch',
        log_momentum=False,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir='logs/',
    )

    # Initialize model
    model = HistologySegmentationModel(
        arch=cfg.architecture,
        encoder_name=cfg.encoder,
        in_channels=3,
        classes=cfg.classes,
    )

    # Initialize and tun trainer
    trainer = pl.Trainer(
        devices=-1,
        accelerator=cfg.device,
        max_epochs=cfg.epochs,
        logger=tb_logger,
        callbacks=[
            lr_monitor,
            checkpoint,
        ],
        enable_checkpointing=True,
        log_every_n_steps=cfg.batch_size,
        default_root_dir=model_dir,
    )
    trainer.fit(
        model,
        datamodule=oct_data_module,
    )


if __name__ == '__main__':
    main()
