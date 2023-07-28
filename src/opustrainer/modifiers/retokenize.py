from typing import List, Protocol, Dict, NamedTuple, TypeVar, Callable, Union, Tuple, Optional, Any
from itertools import count

from opustrainer.types import Pair, TokenList, TokenMapping, Tokenizer, Detokenizer
from opustrainer.alignments import parse_alignments, format_alignments
from opustrainer.tokenizers import make_tokenizer, make_detokenizer
from opustrainer.modifiers import Modifier
from opustrainer import logger


def overlaps(r1:slice, r2:slice) -> bool:
    """True if slice 1 (partially or fully) overlaps with slice 2."""
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

        #TODO: This can be done much more efficiently
        for i, old_token_span in enumerate(old_token_spans):
            for j, new_token_span in enumerate(new_token_spans):
                if overlaps(old_token_span, new_token_span):
                    old_to_new_mapping[i].append(j)

        return detokenized, new_tokens, old_to_new_mapping


def make_retokenizer(spec:Dict[str,str]) -> Retokenizer:
    return Retokenizer(
        detokenizer=make_detokenizer(spec.get('detokenize', 'spaces')),
        tokenizer=make_tokenizer(spec.get('tokenize', 'spaces'))
    )


def remap_alignment_pairs(src_mapping:TokenMapping, trg_mapping:TokenMapping, alignments:List[Pair]) -> List[Pair]:
    """Will recalculate the alignment pairs to match a new tokenization scheme
    according to the updated mappings for the source and target side of the
    sentence pair.
    
    E.g. if you have
    source-mapping: [0 => [3,4], 1 => [5]],
    target-mapping: [0 => [0], 1 => [1]]
    alignments:     [(0,1), (1,1)]
    it will return  [
        (3,1), (4,1), # the [0 => [3,4]] mapping
        (5,1)         # the [1 => [5]] mapping
    ]
    """
    remapped = set()
    for old_src_idx, old_trg_idx in alignments:
        for src_idx in src_mapping[old_src_idx]:
            for trg_idx in trg_mapping[old_trg_idx]:
                remapped.add(Pair(src_idx, trg_idx))
    return sorted(remapped)


class RetokenizeModifier(Modifier):
    """Retokenizes the input line, fixing up the alignments *but giving you the detokenized text*.
    The probability argument is ignored. Most of this functionality is already built into the Tags
    placeholder.

    Default detokenizer and tokenizer are `spaces`, which just splits and joins the tokens with a
    space in between. Other options are `moses:{lang}` for Moses detokenizer and tokenizer, and
    `spm` for sentencepiece.

    `spm` is only available as a tokenize option at this moment since other modifiers won't know
    how to deal with spm input anyway.
    
    Usage:
    ```yaml
    modifiers:
    - Retokenize: 0
      src:
        detokenize: moses:en
        tokenize: spm:path/to/vocab.spm
      trg:
        detokenize: moses:zh
        tokenize: spm:path/to/vocab.spm

    ```
    """

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
        remapped_pairs = remap_alignment_pairs(src_mapping, trg_mapping, pairs)
        return '\t'.join((new_src, new_trg, format_alignments(remapped_pairs)))

