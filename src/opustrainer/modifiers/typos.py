import random
import re
from typing import Dict, Literal, Tuple

import typo

from opustrainer.modifiers import Modifier


def random_space_with_alignment(action:Literal['random_space', 'skipped_space'], strval:str, alignments:str) -> Tuple[str, str]:
    """Special version of typo's random_space and skipped_space that also
    updates alignment info.
    
    action: add | remove
      whether to add or remove a random space

    strval: str
      input text

    alignments: str
      string of space split m-n pairs

    """
    # all the locations where there are non-space characters.
    locations = [m.start() for m in re.finditer(r'\S', strval)]
    
    if len(locations) == 0:
        return strval, alignments

    # Select character after which to add a space
    char_index = locations[random.randint(0, len(locations) - 1)]

    # Figure out which word that character falls in
    word_index = sum(1 for _ in re.finditer(r'\s+', strval[:char_index]))

    # Insert space
    if action == 'random_space':
        strval = strval[:char_index] + ' ' + strval[char_index:]
    else:
        strval = strval[:char_index-1] + strval[char_index:]

    # Fix up alignments
    fixed_alignments = []
    for alignment in alignments.split(' '):
        # Splits the a-b pairs into tuples
        src, trg = [int(index) for index in alignment.split('-', maxsplit=1)]

        # Alignments before the introduced space stay as-is. Intentionally, if
        # the mapping is about word_index itself, we apply both to duplicate
        # the mapping.
        if src <= word_index:
            fixed_alignments.append(f'{src}-{trg}')
        
        # Alignments after the space are shifted by 1
        if action == 'random_space' and src >= word_index \
           or action == 'skipped_space' and src > word_index:
            src += 1 if action == 'random_space' else -1
            fixed_alignments.append(f'{src}-{trg}')

    return strval, ' '.join(fixed_alignments)


class TypoModifier(Modifier):
    # modifier name, and probability it is applied on a considered
    # sentence. Each modifier can either be applied once or not at all
    # for a considered sentence. The default probability for each is 10%.
    modifiers = {
        'char_swap':     0.1,
        'missing_char':  0.1,
        'extra_char':    0.1,
        'nearby_char':   0.1,
        'similar_char':  0.1,
        'skipped_space': 0.1,
        'random_space':  0.1,
        'repeated_char': 0.1,
        'unichar':       0.1,
    }

    column: int

    probabilities: Dict[str,float]

    def __init__(self, probability:float, **probabilities:float):
        """
        Apply typo modifiers to the input. If no specific typo modifiers are
        mentioned, it will default to applying them all with a 0.1 probability
        each. If modifiers are mentioned in the configuration, only the
        modifiers mentioned will be used. All probabilities have to be in the
        0.0 .. 1.0 range.

        args:
            probability: float
                probability a line will be modified

            char_swap: float
                Swaps two random consecutive word characters in the string.

            missing_char: float
                Skips a random word character in the string.

            extra_char: float
                Adds an extra, keyboard-neighbor, letter next to a random word character.

            nearby_char: float
                Replaces a random word character with keyboard-neighbor letter.

            similar_char: float
                Replaces a random word character with another visually similar character.

            skipped_space: float
                Skips a random space from the string.

            random_space: float
                Adds a random space in the string.

            repeated_char: float
                Repeats a random word character.

            unichar: float
                Replaces a random consecutive repeated letter with a single letter.
        """
        super().__init__(probability)

        for mod, mod_prob in probabilities.items():
            if mod not in self.modifiers:
                raise ValueError(f'Unknown typo modifier: {mod}')
            if mod_prob < 0.0 or mod_prob > 1.0:
                raise ValueError(f'Typo modifier {mod} has a probability out of the 0.0..1.0 range')

        self.probabilities = probabilities or self.modifiers

    def __call__(self, line:str) -> str:
        fields = line.split("\t")

        # TODO: The StrErrer constructor calls random.seed(None), which isn't
        # great for reproducibility. Not sure whether getrandbits() is a good
        # workaround though.
        data = typo.StrErrer(fields[0], seed=random.getrandbits(32))

        has_alignment_info = len(fields) > 2

        for modifier, probability in self.probabilities.items():
            if probability > random.random():
                # Introducing spaces with alignment information is a problem.
                if has_alignment_info and modifier in ('random_space', 'skipped_space'):
                    data.result, fields[2] = random_space_with_alignment(modifier, data.result, fields[2])
                else:
                    wordcount = len(data.result.split(' '))
                    getattr(data, modifier)()
                    assert len(data.result.split(' ')) == wordcount or not has_alignment_info, f'Modifier {modifier} changed the word count while alignment info was not updated'


        fields[0] = data.result

        return "\t".join(fields)
