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
        base_task_id='b3aa2d33ec3344329413549955c621b2',  # TODO:
        # Set the hyperparameter range to search
        hyper_parameters=[
            DiscreteParameterRange(
                name='General/encoder',
                values=cfg.encoder,  # TODO: keep the values of the arguments in one place
            ),
            DiscreteParameterRange(
                name='General/optimizer',
                values=cfg.optimizer,  # TODO: keep the values of the arguments in one place
            ),
            UniformParameterRange(
                name='General/lr',  # TODO: LRs = [0.00001, 0.0001, 0.001]
                min_value=0.00002,  # TODO: keep the values of the arguments in one place
                max_value=0.0004,  # TODO: keep the values of the arguments in one place
                step_size=0.00002,  # TODO: keep the values of the arguments in one place
            ),
            DiscreteParameterRange(
                name='General/input_size',  # TODO: start = 256, step = 128, stop = 1024 (depends on the amount of GPU memory)
                values=[224, 256, 448],  # TODO: keep the values of the arguments in one place
            ),
            DiscreteParameterRange(
                name='General/dropout',  # TODO: start = 0, step = 0.05, stop = 0.5
                values=[
                    0.0,
                    0.1,
                    0.25,
                    0.35,
                    0.5,
                ],  # TODO: keep the values of the arguments in one place
            ),
        ],
        # Set the target metric that we want to maximize or minimize
        objective_metric_title='val',  # TODO: move this value to tune.yaml and use it as parameter
        objective_metric_series='f1',  # TODO: move this value to tune.yaml and use it as parameter
        objective_metric_sign='max',  # TODO: move this value to tune.yaml and use it as parameter
        # Set a search strategy optimizer
        optimizer_class=OptimizerBOHB,
        # Configure optimization parameters
        execution_queue='default',
        max_number_of_concurrent_tasks=1,  # TODO: Why did you set it to 1?
        total_max_jobs=50,
        min_iteration_per_job=1500,
        max_iteration_per_job=5000,
    )
    optimizer.start_locally()
    optimizer.wait()
    optimizer.stop()


if __name__ == '__main__':
    main()
