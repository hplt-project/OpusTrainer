# This file contains merge modifier and noise modifier
import random
from opustrainer.modifiers import Modifier

import random
from typing import List, Sequence, Union
from opustrainer.modifiers import Modifier
from opustrainer.alignments import format_alignments, parse_alignments, Pair

def merge_sents(inputs: List[str]) -> str:
    """Merges n sentences together, fixing up their alignments"""
    srcs: List[List[str]] = [x.split('\t')[0].split() for x in inputs]
    trgs: List[List[str]] = [x.split('\t')[1].split() for x in inputs]
    align_txt: Union[str, None] = None
    if len(inputs[0].split('\t')) > 2:
        aligns: List[List[Pair]] = [parse_alignments(x.split('\t')[2].strip()) for x in inputs]

        add_src = len(srcs[0])
        add_trg = len(trgs[0])
        for i in range(1, len(srcs)):
            for j in range(len(aligns[i])):
                aligns[i][j] = Pair(aligns[i][j][0] + add_src, aligns[i][j][1] + add_trg)
            add_src = add_src + len(srcs[i])
            add_trg = add_trg + len(trgs[i])

        align_txt = format_alignments([item for sublist in aligns for item in sublist])

    srcs_txt: str = " ".join([x.split('\t')[0] for x in inputs])
    trgs_txt: str = " ".join([x.split('\t')[1] for x in inputs])

    if align_txt is not None:
        return srcs_txt + '\t' + trgs_txt + '\t' + align_txt
    else:
        return srcs_txt + '\t' + trgs_txt
        
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
    min_lines_merge: int
    max_lines_merge: int
    def __init__(self, probability: float=0.0, min_lines_merge: int=2, max_lines_merge: int=4):
        super().__init__(probability)
        self.min_lines_merge = min_lines_merge
        self.max_lines_merge = max_lines_merge

    def __call__(self, batch:List[str]) -> Sequence[str]:
        newbatch: List[str] = []
        # Identify merging candidates and their lengths
        prev_end = -1
        for i in range(len(batch)):
            if i < prev_end:
                continue
            elif self.probability > random.random():
                merge_end = i + random.randint(self.min_lines_merge, self.max_lines_merge)
                prev_end = merge_end
                merge_batch: str = merge_sents(batch[i:merge_end])
                newbatch.append(merge_batch)
            else:
                newbatch.append(batch[i])

        return newbatch
