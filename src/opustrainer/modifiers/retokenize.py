from typing import List, Dict, NamedTuple, Tuple, Iterable

from opustrainer.types import Pair, TokenList, TokenMapping, Tokenizer, Detokenizer
from opustrainer.alignments import parse_alignments, format_alignments
from opustrainer.tokenizers import make_tokenizer, make_detokenizer
from opustrainer.modifiers import Modifier
from opustrainer import logger


def slice_cmp(r1:slice, r2:slice) -> int:
    """Compare how two slices relate to each other.
       Returns -1 if r1 is before r2,
                0 if r1 and r2 overlap
                1 if r1 is after r2
    """
    if r1.stop <= r2.start:
        return -1
    elif r1.start >= r2.stop:
        return 1
    else:
        return 0


class Retokenizer(NamedTuple):
    detokenizer: Detokenizer
    tokenizer: Tokenizer

    def retokenize(self, tokens:TokenList) -> Tuple[str,TokenList,TokenMapping]:
        detokenized, old_token_spans = self.detokenizer.detokenize(tokens)
        new_tokens, new_token_spans = self.tokenizer.tokenize(detokenized)

        old_to_new_mapping = [[] for _ in range(len(old_token_spans))]

        prev_j = 0
        for i, old_token_span in enumerate(old_token_spans):
            # it is possible for ICU tokenizer whitespace token, return empty list
            if old_token_span is None:
                continue

            for j, new_token_span in enumerate(new_token_spans[prev_j:], start=prev_j):
                prev_j = j
                overlap = slice_cmp(old_token_span, new_token_span)
                if overlap < 0: # old_token_span is before new_token_span
                    break # skip to next old_token_span
                elif overlap > 0: # old_token_span is after new_token_span
                    continue # skip forward to next new_token_span
                else:
                    old_to_new_mapping[i].append(j) # overlap!

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
    source-mapping: [0 => [3,4], 1 => [5], 2 => []],
    target-mapping: [0 => [0], 1 => [1], 2 => []]
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

    def __call__(self, batch:List[str]) -> Iterable[str]:
        for line in batch:
            try:
                src, trg, alignments = line.split('\t')
                src_tokens = src.split()
                trg_tokens = trg.split()
                pairs = parse_alignments(alignments, src_tokens, trg_tokens)
                new_src, _, src_mapping = self.src.retokenize(src_tokens)
                new_trg, _, trg_mapping = self.trg.retokenize(trg_tokens)
                remapped_pairs = remap_alignment_pairs(src_mapping, trg_mapping, pairs)
                yield '\t'.join((new_src, new_trg, format_alignments(remapped_pairs)))
            except Exception as exc:
                logger.log(f'Exception while processing line, skipping line: {exc!r}', 'WARNING')
