from typing import Optional, List
from opustrainer.types import Pair, TokenList


def parse_alignments(pairs:str, src_tokens:Optional[TokenList]=None, trg_tokens:Optional[TokenList]=None) -> List[Pair]:
    pairs = [
        Pair(int(a), int(b)) for a, b in
        (pair.split('-', maxsplit=1) for pair in pairs.split())
    ]

    if src_tokens is not None and trg_tokens is not None:
        invalid_pairs = [
            pair
            for pair in pairs
            if pair.src < 0 or pair.src >= len(src_tokens)
            or pair.trg < 0 or pair.trg >= len(trg_tokens)
        ]
        if invalid_pairs:
            raise ValueError('Out-of-bound alignment pairs: ' + ' '.join(map(repr, invalid_pairs)))

    return pairs


def format_alignments(pairs:List[Pair]) -> str:
    return ' '.join(f'{pair.src}-{pair.trg}' for pair in pairs)
