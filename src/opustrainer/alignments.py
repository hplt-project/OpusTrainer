from typing import Optional, List, Union
from opustrainer.types import Pair, TokenList


def parse_alignments(input_pairs:str, src_tokens:Optional[TokenList]=None, trg_tokens:Optional[TokenList]=None) -> List[Pair]:
    """Parses `1-2 3-4` into `[Pair(src=1,trg=2), Pair(src=3,trg=4)]`. If `src_tokens` and
    `trg_tokens` are supplied, it will also check that the indices are not out of bounds."""
    pairs: List[Pair] = [
        Pair(int(a), int(b)) for a, b in
        (pair.split('-', maxsplit=1) for pair in input_pairs.split())
    ]

    if src_tokens is not None and trg_tokens is not None:
        for pair in pairs:
            if pair.src < 0 or pair.src >= len(src_tokens) \
            or pair.trg < 0 or pair.trg >= len(trg_tokens):
                raise ValueError('Out-of-bound alignment pairs')

    return pairs


def format_alignments(pairs:List[Pair]) -> str:
    """Opposite of `parse_alignments`, turns a list of alignments back into the `a-b c-d ...` string
    format that most alignment tools expect."""
    return ' '.join(f'{pair.src}-{pair.trg}' for pair in pairs)
