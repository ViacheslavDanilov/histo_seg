import datetime
import gc
import logging
import os

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models.smp.dataset import HistologyDataModule
from src.models.smp.model import HistologySegmentationModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='tune',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    os.environ['WANDB_API_KEY'] = '0a94ef68f2a7a8b709671d6ef76e61580d20da7f'
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    sweep_config = {
        'method': 'bayes',
        'metric': {'name': f'{cfg.metric_type}/{cfg.metric_name}', 'goal': cfg.metric_sign},
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 2,
            'max_iter': 50,
        },  # 8 (16/2), 4 (16/2/2)
        'parameters': {
            'classes': {'value': list(cfg.classes)},
            'architecture': {'value': cfg.architecture},
            'batch_size': {'value': cfg.batch_size},
            'cuda_num': {'value': list(cfg.cuda_num)},
            # Variable hyperparameters
            'input_size': {
                'values': list(
                    range(cfg.input_size_min, cfg.input_size_max + 1, cfg.input_size_step),
                ),
            },
            'optimizer': {'values': list(cfg.optimizer)},
            'lr': {'values': list(cfg.learning_rate)},
            'encoder': {'values': list(cfg.encoder)},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_config, entity='vladislavlaptev', project=cfg.project_name)
    wandb.agent(sweep_id=sweep_id, function=tune, count=200)

    # If the tuning is interrupted, use a specific sweep_id to keep tuning on the next call
    # wandb.agent(sweep_id='3t2kelpq', function=tune, count=200, entity='vladislavlaptev', project=cfg.project_name)

    print('\n\033[92m' + '-' * 100 + '\033[0m')
    print('\033[92m' + 'Tuning has finished!' + '\033[0m')
    print('\033[92m' + '-' * 100 + '\033[0m')


def tune(config=None):
    with wandb.init(config=config):
        config = wandb.config
        run_name = wandb.run.name
        print('\033[92m' + '\n********** Run: {:s} **********\n'.format(run_name) + '\033[0m')

        today = datetime.datetime.today()

        task_name = f'histology_segmentation_{today.strftime("%d%m_%H%M%S")}'
        model_dir = os.path.join('models', f'{task_name}')

        callbacks = [
            LearningRateMonitor(
                logging_interval='epoch',
                log_momentum=False,
            ),
        ]

        os.makedirs(f'{model_dir}')

        oct_data_module = HistologyDataModule(
            input_size=config.input_size,
            classes=config.classes,
            batch_size=config.batch_size,
            num_workers=os.cpu_count(),
            data_dir='data/final',
        )
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir='logs/',
        )

        # Initialize model
        model = HistologySegmentationModel(
            arch=config.architecture,
            encoder_name=config.encoder,
            optimizer_name=config.optimizer,
            in_channels=3,
            classes=config.classes,
            model_name=task_name,
            lr=config.lr,
            save_img_per_epoch=False,
        )

        # Initialize and run trainer
        trainer = pl.Trainer(
            devices=config.cuda_num,
            accelerator='cuda',
            max_epochs=50,
            logger=tb_logger,
            callbacks=callbacks,
            enable_checkpointing=True,
            log_every_n_steps=4,
            default_root_dir=model_dir,
        )

        try:
            trainer.fit(
                model,
                datamodule=oct_data_module,
            )
        except Exception:
            print('Run status: CUDA out-of-memory error or HyperBand stop')
        else:
            print('Run status: Success')
        finally:
            print('Reset memory and clean garbage')
            gc.collect()
            torch.cuda.empty_cache()
        wandb.join()


if __name__ == '__main__':
    main()
