from abc import ABC, abstractmethod

from fast_dynamic_batcher.models import Task


class InferenceModel(ABC):
    @abstractmethod
    def infer(self, tasks: list[Task]) -> list[Task]: ...
