import random
from typing import List, Iterable

from opustrainer.alignments import format_alignments, Pair
from opustrainer.modifiers import Modifier
from opustrainer.modifiers.placeholders import get_random_unicode_words


class NoiseModifier(Modifier):
    """Adds additional sentence pairs entirely made of noise during training.
    Nonsensical string on the source and on the target.

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

    def __init__(self, probability: float, min_word_length: int=2,
                 max_word_length: int=5, max_words: int=6):
        super().__init__(probability)
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.max_words = max_words

    def __call__(self, batch:List[str]) -> Iterable[str]:
        """Generates a random noise line"""
        for line in batch:
            if self.probability > random.random():
                tokens = get_random_unicode_words(self.min_word_length, self.max_word_length, self.max_words)
                fake_line = " ".join(tokens) + "\t" + " ".join(tokens)
                
                # If there's a third field in the original line, we will also
                # add alignment information to this fake line. We assume 1-1
                # mapping.
                if line.count('\t') >= 2:
                    alignments = [Pair(i, i) for i in range(len(tokens))]
                    fake_line += "\t" + format_alignments(alignments)
                
                yield fake_line
            
            # Always also yield the original line
            yield line
