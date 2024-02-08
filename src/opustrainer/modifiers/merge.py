import random
from typing import List, Iterable, Optional
from itertools import accumulate, chain

from opustrainer.modifiers import Modifier
from opustrainer.alignments import format_alignments, parse_alignments, Pair
from opustrainer.tokenizers import SpaceDetokenizer, SpaceTokenizer


def merge_sents(inputs: List[str]) -> str:
    """Merges n sentences together, fixing up their alignments"""
    rows = [line.split('\t') for line in inputs]

    input_tokens = (
        [row[0].split() for row in rows], # src
        [row[1].split() for row in rows], # trg
    )

    # src and trg sentences, merged
    output_cols = [" ".join(chain(*side)) for side in input_tokens]

    # If all rows have alignment info, add a merged alignment column
    if all(len(row) > 2 for row in rows):
        # Alignments as Pairs per input sentence
        alignments = [parse_alignments(row[2]) for row in rows]

        # Offsets for where alignments per input sentence should start
        offsets = tuple(
            list(accumulate((len(sentence) for sentence in side), initial=0))
            for side in input_tokens
        )

        # New alignment pairs that have been offset to account for the sentences
        # that would precede it.
        joined_alignments = [
            Pair(p.src + offsets[0][i], p.trg + offsets[1][i])
            for i, sentence_pairs in enumerate(alignments)
            for p in sentence_pairs
        ]

        output_cols.append(format_alignments(joined_alignments))

    return "\t".join(output_cols)


class MergeModifier(Modifier):
    """Randomly merges up to n lines into one
    
        Usage:
       ```yaml
       modifiers:
       - Merge: 0.01
         min_lines: 2
         max_lines: 4
        ```
    """
    min_lines: int
    max_lines: int

    def __init__(self, probability: float, min_lines: int=2, max_lines: int=4):
        super().__init__(probability)
        self.min_lines = min_lines
        self.max_lines = max_lines

    def __call__(self, batch:List[str]) -> Iterable[str]:
        i = 0
        while i < len(batch):
            if self.probability > random.random():
                merge_size = random.randint(self.min_lines, self.max_lines)
                yield merge_sents(batch[i:i+merge_size])
                i += merge_size
            else:
                yield batch[i]
                i += 1
