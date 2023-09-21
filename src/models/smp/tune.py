import logging
import os
from typing import List

import hydra
from clearml import Task
from clearml.automation import DiscreteParameterRange, HyperParameterOptimizer
from clearml.automation.hpbandster import OptimizerBOHB
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_input_size(
    min_value: int,
    step_size: int,
    max_value: int,
) -> List[int]:
    return range(min_value, max_value, step_size)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='tune',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Initialize ClearML task and log hyperparameters
    Task.init(
        project_name=cfg.project_name,
        task_name='HPO',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
        auto_connect_frameworks={'tensorboard': True, 'pytorch': True},
    )

    optimizer = HyperParameterOptimizer(
        base_task_id='04cbe94d39bc4516ba7f389e3950f501',
        hyper_parameters=[
            DiscreteParameterRange(
                name='General/encoder',
                values=cfg.encoder,
            ),
            DiscreteParameterRange(
                name='General/optimizer',
                values=cfg.optimizer,
            ),
            DiscreteParameterRange(
                name='General/lr',
                values=cfg.learning_rate,
            ),
            DiscreteParameterRange(
                name='General/input_size',
                values=get_input_size(
                    min_value=int(cfg.input_size_min),
                    step_size=int(cfg.input_size_step),
                    max_value=int(cfg.input_size_max),
                ),
            ),
        ],
        objective_metric_title=cfg.key_metric_type,
        objective_metric_series=cfg.key_metric_name,
        objective_metric_sign=cfg.key_metric_sign,
        optimizer_class=OptimizerBOHB,
        execution_queue='default',
        max_number_of_concurrent_tasks=1,
        total_max_jobs=50,
        min_iteration_per_job=1500,
        max_iteration_per_job=5000,
    )
    optimizer.start_locally()
    optimizer.wait()
    optimizer.stop()


if __name__ == '__main__':
    main()
