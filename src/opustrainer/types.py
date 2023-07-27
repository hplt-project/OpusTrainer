from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, List, Tuple, Optional, Any, Protocol


TokenList = List[str] # todo: bytes?

TokenMapping = List[List[int]]


class Tokenizer(Protocol):
    """Turns a string into a list of tokens"""
    def tokenize(self, text:str) -> Tuple[TokenList,List[slice]]:
        ...


class Detokenizer(Protocol):
    """Turns a list of tokens into a string"""
    def detokenize(self, tokens:TokenList) -> Tuple[str, List[slice]]:
        ...


class Pair(NamedTuple):
    """Alignment pair between source and target token indices"""
    src:int
    trg:int


class SentencePair(NamedTuple):
    """Semantic representation of a single line from a data source."""
    src: TokenList
    trg: TokenList

    # alignments is an empty list if alignment data is available in the dataset
    # but there are no aligned tokens in this pair. It is None if this dataset
    # does not have alignment info.
    alignments: Optional[List[Pair]]


class Modifier(ABC):
    """Line modifier"""
    probability: float

    def __init__(self, probability:float, **kwargs:Dict[str,Any]):
        self.probability = probability

    def validate(self, context:List['Modifier']) -> None:
        """Opportunity for the modifier to see where in the modifier list it is
        placed and flag any issues to the logger. E.g. if you place a modifier that
        inserts special tokens before an UpperCase modifier, the latter might
        modify those special tokens as well. Here you can shout about that.
        """
        pass

    @abstractmethod
    def __call__(self, line: str) -> str:
        pass
