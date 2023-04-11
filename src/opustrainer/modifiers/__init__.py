from abc import ABC, abstractmethod
from typing import Dict, Any, List


class Modifier(ABC):
    probability: float

    def __init__(self, probability:float, **kwargs:Dict[str,Any]):
        self.probability = probability

    def validate(self, context:List['Modifier']) -> None:
        pass

    @abstractmethod
    def __call__(self, line: str) -> str:
        pass
