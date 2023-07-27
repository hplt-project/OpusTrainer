from typing import List, Protocol, Dict, NamedTuple, TypeVar, Callable, Union, Tuple, Optional, Any
from itertools import count

from opustrainer.types import Pair, TokenList, TokenMapping, Tokenizer, Detokenizer
from opustrainer.alignments import parse_alignments, format_alignments
from opustrainer.tokenizers import make_tokenizer, make_detokenizer
from opustrainer.modifiers import Modifier
from opustrainer import logger


def print_cells(*rows, format:Callable[[Any],str]=repr,file=None) -> None:
    # Convert it all to text
    text_rows = [
        [format(cell) for cell in row]
        for row in rows
    ]
    
    # Calculate column width
    widths = [
        max(len(cell) for cell in column)
        for column in zip(*text_rows)
    ]

    # Print rows
    for row in text_rows:
        for width, cell in zip(widths, row):
            print(f'{{:<{width:d}}} '.format(cell), end='', file=file)
        print(file=file)


def overlaps(r1:slice, r2:slice) -> bool:
    # (a,b), (x,y) = r1, r2
    #      [a    b]             | a < y |  x < b
    # [x y]                 = F |   F   |    T
    #     [x y]             = T |   T   |    T
    #         [x y]         = T |   T   |    T
    #            [x y]      = T |   T   |    T
    #                [x  y] = F |   T   |    F
    return r1.start < r2.stop and r2.start < r1.stop


class Retokenizer(NamedTuple):
    detokenizer: Detokenizer
    tokenizer: Tokenizer

    def retokenize(self, tokens:TokenList) -> Tuple[str,TokenList,TokenMapping]:
        detokenized, old_token_spans = self.detokenizer.detokenize(tokens)
        new_tokens, new_token_spans = self.tokenizer.tokenize(detokenized)

        old_to_new_mapping = [[] for _ in range(len(old_token_spans))]

        # print(f"{detokenized=}")
        # print_cells(new_tokens, new_token_spans)
        # print(f"{new_token_spans=}")

        #TODO: This can be done much more efficiently
        for i, old_token_span in enumerate(old_token_spans):
            for j, new_token_span in enumerate(new_token_spans):
                if overlaps(old_token_span, new_token_span):
                    old_to_new_mapping[i].append(j)

        # for n, old_token_span, new_token_indices in zip(count(), old_token_spans, old_to_new_mapping):
        #   print(f'<{n}>[{detokenized[old_token_span]}]({old_token_span.start},{old_token_span.stop}): ' + ' '.join(
        #     f'<{new_idx}>[{detokenized[new_token_spans[new_idx]]}]({new_token_spans[new_idx].start},{new_token_spans[new_idx].stop})'
        #     for new_idx in new_token_indices
        #   ))
        # print()

        return detokenized, new_tokens, old_to_new_mapping


def make_retokenizer(spec:Dict[str,str]) -> Retokenizer:
    return Retokenizer(
        detokenizer=make_detokenizer(spec.get('detokenize', 'spaces')),
        tokenizer=make_tokenizer(spec.get('tokenize', 'spaces'))
    )


def compute_mapping(src_mapping:TokenMapping, trg_mapping:TokenMapping, alignments:List[Pair]) -> List[Pair]:
    remapped = set()
    for old_src_idx, old_trg_idx in alignments:
        for src_idx in src_mapping[old_src_idx]:
            for trg_idx in trg_mapping[old_trg_idx]:
                remapped.add(Pair(src_idx, trg_idx))
    return sorted(remapped)


class RetokenizeModifier(Modifier):
    src: Retokenizer
    trg: Retokenizer

    def __init__(self, probability: float=0.0, src:dict=dict(), trg:dict=dict()):
        super().__init__(probability) # probability is very much ignored lol.
        self.src = make_retokenizer(src)
        self.trg = make_retokenizer(trg)

    def __call__(self, line:str) -> str:
        src, trg, alignments = line.split('\t')
        src_tokens = src.split()
        trg_tokens = trg.split()
        pairs = parse_alignments(alignments, src_tokens, trg_tokens)
        new_src, new_src_tokens, src_mapping = self.src.retokenize(src_tokens)
        new_trg, new_trg_tokens, trg_mapping = self.trg.retokenize(trg_tokens)
        remapped_pairs = compute_mapping(src_mapping, trg_mapping, pairs)
        return '\t'.join((new_src, new_trg, format_alignments(remapped_pairs)))

