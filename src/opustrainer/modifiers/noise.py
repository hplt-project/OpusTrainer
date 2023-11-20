# This file contains merge modifier and noise modifier
import random
from opustrainer.modifiers import Modifier
from opustrainer.modifiers.placeholders import get_random_unicode_words

import random
from typing import List, Sequence
from opustrainer.modifiers import Modifier

class NoiseModifier(Modifier):
    """Adds noise during training. Nonsensitcal string on the source and on the target

       Usage:
       ```yaml
       modifiers:
       - Noise: 0.01
         min_word_length: 2
         max_word_length: 5
         max_words: 6
        ```
    """
    min_word_length: int
    max_word_length: int
    max_words: int

    def __init__(self, probability: float=0.0, min_word_legnth: int=2,
        max_word_length: int=5, max_words: int=6):
        super().__init__(probability)
        self.min_word_length = min_word_legnth
        self.max_word_length = max_word_length
        self.max_words = max_words

    def __call__(self, batch:List[str]) -> Sequence[str]:
        """Generates a random noise line"""
        # The only problem is that we don't know if the dataset is supposed to have an alignment field
        # or not... A tradeoff is to look at the previous line and see if it has alignment info and then follow that
        # it's not ideal as we might hit a defective line, but oh well...
        ret_batch: List[str] = []
        for line in batch:
            if self.probability > random.random():
                newline: str =  " ".join(get_random_unicode_words(self.min_word_length, self.max_word_length, self.max_words))
                # Check if we have a 3rd field, which we assume is alignment
                if line.count('\t') == 2:
                    # Generate alignments, just in case
                    alignments: str = ""
                    myrange = range(newline.count(' ') + 1)
                    for j in myrange:
                        alignments = alignments + str(j) + '-' + str(j) + " "
                    alignments = alignments[:-1] # remove final space
                    ret_batch.append(newline +'\t' + newline + '\t' + alignments)
                else:
                    ret_batch.append(newline +'\t' + newline)
            ret_batch.append(line)
        return ret_batch
