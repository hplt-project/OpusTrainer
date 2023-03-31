from typing import Dict

import typo

from opustrainer.modifiers import Modifier


class TypoModifier(Modifier):
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

    def __init__(self, probability:float, column:int = 0, **probabilities:Dict[str,float] = {}):
        """
        Apply typo modifiers to the input. If no specific typo modifiers are
        mentioned, it will default to applying them all with a 0.1 probability
        each. If modifiers are mentioned in the configuration, only the
        modifiers mentioned will be used. All probabilities have to be in the
        0.0 .. 1.0 range.

        args:
            probability: float
                probability a line will be modified

            column: int
                column to apply modifiers to. By default it is the first column.

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

        self.column = column

        for modifier, probability in probabilities.items():
            if modifier not in self.modifiers:
                raise ValueError(f'Unknown typo modifier: {modifier}')
            if probability < 0.0 or probability > 1.0:
                raise ValueError(f'Typo modifier {modifier} has a probability out of the 0.0..1.0 range')

        self.probabilities = probabilities or self.modifiers

    def __call__(self, line:str) -> str:
        fields = line.split("\t")

        # TODO: The StrErrer constructor calls random.seed(None), which isn't
        # great for reproducibility. Not sure whether getrandbits() is a good
        # workaround though.
        data = typo.StrErrer(fields[self.column], seed=random.getrandbits(32))

        for modifier, probability in self.probabilities.items():
            if probability > random.random():
                getattr(data, modifier)()

        fields[self.column] = data.result

        return "\t".join(fields)
