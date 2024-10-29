from abc import ABC, abstractmethod

from fast_dynamic_batcher.models import Task


class InferenceModel(ABC):
    """
    The InferenceModel abstract class is used as a template in the DynBatcher. An implementation of it must be passed to it.
    The __main__ method of an implementation is used to load the machine learning model in the worker process created by the DynBatcher.
    The infer method is used to process the batched inputs.

    :param ABC: Dynamic base class
    """

    @abstractmethod
    def infer(self, tasks: list[Task]) -> list[Task]:
        """
        Abstract method that is used by the DynBatcher to process a batch of inputs. Use the tasks to create a batch for your machine learning model.


        :param tasks: A list of Tasks. The content of each task contains the input passed to the DynBatcher for batch processing.
        :type tasks: list[Task]
        :return: The outputs of the machine learning model as a list of tasks. The tasks should have the same ids as the inputs and the content of the task should be the processed content of the corresponding input task.
        :rtype: list[Task]
        """
        ...
