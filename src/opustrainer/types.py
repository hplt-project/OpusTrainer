from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, List, Iterable, Tuple, Optional, Any, Protocol


# List of tokens/words according to some tokenization scheme
TokenList = List[str] # todo: bytes?

# List of string offsets that accompanies a `TokenList` telling you where each
# token is found in the original string.
TokenSpanList = List[slice]

# Mapping from `old token index` => [new token indices].
TokenMapping = List[List[int]]


class Tokenizer(Protocol):
    """Turns a string into a list of tokens and slices. Each token has a slice
    at the same offset that describes where in `text` that token is found."""
    def tokenize(self, text:str) -> Tuple[TokenList,TokenSpanList]:
        ...


class Detokenizer(Protocol):
    """Turns a list of tokens into a string. The list of slices returned tells
    you where each of the input tokens is found in the detokenized string."""
    def detokenize(self, tokens:TokenList) -> Tuple[str, TokenSpanList]:
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
    def __call__(self, batch: List[str]) -> Iterable[str]:
        pass
