import random
import re
from typing import Dict, Tuple, List, Iterable

import typo

from opustrainer.modifiers import Modifier


# Monkey patch typo to not try to find the nearest digit for
# non-digit numerals.
# See https://github.com/hplt-project/OpusTrainer/issues/40
# See https://github.com/ranvijaykumar/typo/issues/3
from  typo.keyboardlayouts import en_default
_get_random_neighbor = en_default.get_random_neighbor

def skip_non_digit_decimals(char, seed=None):
    if char.isdecimal() and char not in en_default.NEIGHBORINGNUMPADDIGITS:
        return char
    return _get_random_neighbor(char, seed)

typo.keyboardlayouts.en_default.get_random_neighbor = skip_non_digit_decimals


def add_random_space_with_alignment(
    string: str, alignment_str: str, include_edges=False
) -> Tuple[str, str]:
    """Add a space to a random point in the string"""
    possible_insertions = [m.start() for m in re.finditer(r"\S", string)]

    if include_edges:
        possible_insertions += [len(string)]  # also at end
    else:
        possible_insertions = possible_insertions[1:]  # not at beginning

    if not possible_insertions:
        return string, alignment_str

    index = random.choice(possible_insertions)
    augmented_string = string[:index] + " " + string[index:]

    # Find positions of space-like segments that would not cause de-alignment
    space_spans = list(re.finditer(r"^\s*|\s+|$", string[: index + 1]))
    keep_alignment_indices = [m.end() for m in space_spans]
    word_index = len(keep_alignment_indices) - 2

    augmented_alignment_str = alignment_str
    if index not in keep_alignment_indices:
        assert len(augmented_string.split()) - len(string.split()) == 1
        augmented_alignment = []

        for align in alignment_str.split():
            src, trg = [int(index) for index in align.split("-", maxsplit=1)]
            if src <= word_index:
                augmented_alignment.append(f"{src}-{trg}")
            if src >= word_index:
                augmented_alignment.append(f"{src+1}-{trg}")

        augmented_alignment_str = " ".join(augmented_alignment)
    return augmented_string, augmented_alignment_str


def skip_random_space_with_alignment(
    string: str, alignment_str: str, include_edges=False
) -> Tuple[str, str]:
    possible_removals = re.finditer(r"(?<!^)\s(?!$)", string)

    if include_edges:
        possible_removals = re.finditer(r"\s", string)
    possible_removals = [m.start() for m in possible_removals]

    if not possible_removals:
        return string, alignment_str

    index = random.choice(possible_removals)
    augmented_string = string[:index] + string[index + 1 :]

    word_spans = list(re.finditer(r"(?<=\S)\s\S", string))
    word_index = None
    for i, match in enumerate(word_spans):
        if index == match.start():
            word_index = i
            break

    augmented_alignment_str = alignment_str
    if word_index is not None:
        augmented_alignment = []
        for align in alignment_str.split():
            src, trg = [int(index) for index in align.split("-", maxsplit=1)]
            if src <= word_index:
                augmented_alignment.append(f"{src}-{trg}")
            else:
                augmented_alignment.append(f"{src-1}-{trg}")
        augmented_alignment_str = " ".join(augmented_alignment)

    return augmented_string, augmented_alignment_str


def missing_char_with_alignment(string: str, alignment_str: str):
    possible_removals = [m.start() for m in re.finditer(r"\S", string)]

    # Do not remove only possible char
    if len(possible_removals) <= 1:
        return string, alignment_str

    index = random.choice(possible_removals)
    augmented_string = string[:index] + string[index + 1 :]

    word_spans = list(re.finditer(r"\S+", string))

    # Find word_index if a single-char word was removed
    word_index = None
    for i, match in enumerate(word_spans):
        start, end = match.span()
        if index > end:
            continue
        if start <= index:
            if end - start == 1:
                word_index = i
            break

    augmented_alignment_string = alignment_str
    if word_index is not None:
        augmented_alignment = []
        # Reassign alignment to a random-neighbour
        neighbours = []
        if word_index != 0:
            neighbours.append(-1)
        if word_index != len(word_spans):
            neighbours.append(0)
        r = random.choice(neighbours)

        for align in alignment_str.split():
            src, trg = [int(index) for index in align.split("-", maxsplit=1)]
            if src == word_index:
                augmented_alignment.append(f"{src+r}-{trg}")
            else:
                offset = 0 if src < word_index else -1
                augmented_alignment.append(f"{src+offset}-{trg}")
        augmented_alignment_string = " ".join(augmented_alignment)

    return augmented_string, augmented_alignment_string

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

    def __call__(self, batch: List[str]) -> Iterable[str]:
        for line in batch:
            yield self.apply(line) if self.probability > random.random() else line

    def apply(self, line:str) -> str:
        fields = line.split("\t")

        # TODO: The StrErrer constructor calls random.seed(None), which isn't
        # great for reproducibility. Not sure whether getrandbits() is a good
        # workaround though.
        data = typo.StrErrer(fields[0], seed=random.getrandbits(32))

        has_alignment_info = len(fields) > 2

        for modifier, probability in self.probabilities.items():
            if probability > random.random():
                # Introducing spaces with alignment information is a problem.
                if has_alignment_info and modifier in (
                    "random_space",
                    "skipped_space",
                    "missing_char",
                ):
                    if modifier == "random_space":
                        m_func = add_random_space_with_alignment
                    elif modifier == "skipped_space":
                        m_func = skip_random_space_with_alignment
                    else:  # modifier == "missing_char":
                        m_func = missing_char_with_alignment

                    data.result, fields[2] = m_func(data.result, fields[2])
                else:
                    wordcount = len(data.result.split())
                    getattr(data, modifier)()
                    assert len(data.result.split()) == wordcount or not has_alignment_info, f'Modifier {modifier} changed the word count while alignment info was not updated'


        fields[0] = data.result

        return "\t".join(fields)
