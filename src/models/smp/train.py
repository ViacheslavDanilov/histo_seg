import datetime
import logging
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # TODO: Is it possible to move this line to the end of the import block?

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
    task_name = f'{cfg.architecture}_{cfg.encoder}_{today.strftime("%d%m_%H%M")}'
    model_dir = os.path.join('models', f'{task_name}')
    os.makedirs(f'{model_dir}/images_per_epoch')
    task = Task.init(
        project_name=cfg.project_name,
        task_name=task_name,
        reuse_last_task_id=False,
        auto_connect_frameworks={'tensorboard': True, 'pytorch': True},
    )
    # Initialize ClearML task and log hyperparameters
    hyperparameters = {
        'architecture': cfg.architecture,
        'encoder': cfg.encoder,
        'input_size': cfg.input_size,
        'classes': list(cfg.classes),
        'num_classes': len(cfg.classes),
        'batch_size': cfg.batch_size,
        'loss': cfg.loss,
        # TODO: add optimizer
        'lr': cfg.lr,
        'dropout': cfg.dropout,
        'epochs': cfg.epochs,
        'device': cfg.device,
        'data_dir': cfg.data_dir,
    }
    hyperparameters = task.connect(hyperparameters)
    task.set_parameters(hyperparameters)
    task.add_tags(
        [
            f'arch: {hyperparameters["architecture"]}',
            f'encd: {hyperparameters["encoder"]}',
            f'opt: {hyperparameters["optimizer"]}',
            f'drp: {hyperparameters["dropout"]}',
            f'lr: {hyperparameters["lr"]}',
            f'inp: {hyperparameters["input_size"]}x{hyperparameters["input_size"]}',
            f'bs: {hyperparameters["batch_size"]}',
        ],
    )

    # Initialize data module
    oct_data_module = HistologyDataModule(
        input_size=hyperparameters['input_size'],
        classes=cfg.classes,
        batch_size=hyperparameters['batch_size'],
        num_workers=os.cpu_count(),
        data_dir=cfg.data_dir,
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
        arch=hyperparameters['architecture'],
        encoder_name=hyperparameters['encoder'],
        optimizer_name=hyperparameters['optimizer'],
        dropout=hyperparameters['dropout'],
        in_channels=3,
        classes=cfg.classes,
        model_name=task_name,
        lr=hyperparameters['lr'],
    )  # TODO: parameter 'save_img_per_epoch' unfilled

    # Initialize and tun trainer
    trainer = pl.Trainer(
        devices=-1,
        accelerator=cfg.device,
        max_epochs=hyperparameters['epochs'],
        logger=tb_logger,
        callbacks=[
            lr_monitor,
            checkpoint,
        ],
        enable_checkpointing=True,
        log_every_n_steps=hyperparameters['batch_size'],
        default_root_dir=model_dir,
    )
    trainer.fit(
        model,
        datamodule=oct_data_module,
    )
    task.upload_artifact(
        name='Metrics',
        artifact_object=f'{model_dir}/metrics.csv',
    )
    task.close()


if __name__ == '__main__':
    main()
