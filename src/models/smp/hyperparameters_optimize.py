import logging
import os

import hydra
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange,
    HyperParameterOptimizer,
    UniformParameterRange,
)
from clearml.automation.hpbandster import OptimizerBOHB
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='hyper_parameter_optimization',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')
    # Initialize ClearML task and log hyperparameters
    Task.init(
        project_name=cfg.project_name,
        task_name='Automatic Hyper-Parameter Optimization',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
        auto_connect_frameworks={'tensorboard': True, 'pytorch': True},
    )

    optimizer = HyperParameterOptimizer(
        base_task_id='b3aa2d33ec3344329413549955c621b2',
        hyper_parameters=[
            DiscreteParameterRange(
                name='General/encoder',
                values=cfg.encoder,
            ),
            DiscreteParameterRange(
                name='General/optimizer',
                values=cfg.optimizer,
            ),
            UniformParameterRange(
                name='General/lr',
                min_value=0.00002,
                max_value=0.0004,
                step_size=0.00002,
            ),
            DiscreteParameterRange(
                name='General/input_size',
                values=[224, 256, 448],
            ),
            DiscreteParameterRange(
                name='General/dropout',
                values=[0.0, 0.1, 0.25, 0.35, 0.5],
            ),
        ],
        # setting the objective metric we want to maximize/minimize
        objective_metric_title='val',
        objective_metric_series='f1',
        objective_metric_sign='max',
        # setting optimizer
        optimizer_class=OptimizerBOHB,
        # configuring optimization parameters
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
