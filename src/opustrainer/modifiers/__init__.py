from abc import ABC, abstractmethod
from typing import Dict, Any


class Modifier(ABC):
    probability: float

    def __init__(self, probability:float, **kwargs:Dict[str,Any]):
        self.probability = probability

    @abstractmethod
    def __call__(self, line: str) -> str:
        pass
