import logging
import os
from pathlib import Path

import hydra
from clearml import Dataset as cl_dataset
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='upload_dataset',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Create the dataset and upload it to ClearML
    dataset = cl_dataset.create(
        dataset_name=Path(cfg.data_dir).name,
        dataset_project=cfg.project_name,
    )
    dataset.add_files(path=cfg.data_dir)
    dataset.upload(
        show_progress=True,
        verbose=True,
    )
    dataset.finalize(verbose=True)


if __name__ == '__main__':
    main()
