import random
from typing import List
from opustrainer.modifiers import Modifier
from opustrainer.modifiers.placeholders import get_random_unicode_string

import random
from typing import List
from opustrainer.modifiers import Modifier


class NoiseModifier(Modifier):
    """Adds noise during training. Nonsensitcal string on the source and on the target

       Usage:
       ```yaml
       modifiers:
       - Noise: 0.01
         min_words: 2
         max_words: 5
         max_word_length: 4
        ```
    """
    min_word_length: int
    max_word_length: int
    max_words: int

    def __init__(self, probability: float=0.0, min_word_legnth: int=2,
        max_word_length: int=5, max_words: int=4):
        super().__init__(probability)
        self.min_word_length = min_word_legnth
        self.max_word_length = max_word_length
        self.max_words = max_words

    def __call__(self, line: str) -> str | None:
        """Generates a random noise line"""
        if self.probability < random.random():
            newline: str =  get_random_unicode_string(self.min_word_length, self.max_word_length, self.max_words)
            # Check if we have a 3rd field, which we assume is alignment
            if line.count('\t') == 2:
                # Generate alignments, in case
                alignments: str = ""
                myrange = range(newline.count(' ') + 1)
                for i in myrange:
                    alignments = alignments + str(i) + '-' + str(i) + " "
                alignments = alignments[:-1] # remove final space
                line = line + '\n' + newline +'\t' + newline + '\t' + alignments
            else:
                line = line + '\n' + newline +'\t' + newline
        return line
