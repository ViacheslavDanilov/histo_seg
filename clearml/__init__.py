""" ClearML open SDK """
from six import PY2

from .datasets import Dataset
from .errors import UsageError
from .logger import Logger
from .model import InputModel, Model, OutputModel
from .storage import StorageManager
from .task import Task
from .version import __version__

TaskTypes = Task.TaskTypes

if not PY2:
    from .automation.controller import PipelineController, PipelineDecorator  # noqa: F401
    from .backend_api import browser_login  # noqa: F401

    __all__ = [
        '__version__',
        'Task',
        'TaskTypes',
        'InputModel',
        'OutputModel',
        'Model',
        'Logger',
        'StorageManager',
        'UsageError',
        'Dataset',
        'PipelineController',
        'PipelineDecorator',
        'browser_login',
    ]
else:
    __all__ = [
        '__version__',
        'Task',
        'TaskTypes',
        'InputModel',
        'OutputModel',
        'Model',
        'Logger',
        'StorageManager',
        'UsageError',
        'Dataset',
    ]
