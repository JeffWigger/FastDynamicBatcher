from abc import ABC, abstractmethod
from typing import Any


class InferenceModel(ABC):
    """
    The InferenceModel abstract class is used as a template in the DynBatcher. An implementation of it must be passed to it.
    The __main__ method of an implementation is used to load the machine learning model in the worker process created by the DynBatcher.
    The infer method is used to process the batched inputs.

    :param ABC: Dynamic base class
    """

    @abstractmethod
    def infer(self, inputs: list[Any]) -> list[Any]:
        """
        Abstract method that is used by the DynBatcher to process a batch of inputs.


        :param inputs: A list of inputs your model can run as a single batch.
        :type inputs: list[Any]
        :return: The outputs of the machine learning model. The position of an output should correspond to the position of the input used to produce it.
        :rtype: list[Any]
        """
        ...
