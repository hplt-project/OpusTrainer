import random
from typing import Callable, Type, List
from functools import partial

from opustrainer.modifiers import Modifier


class LineModifier(Modifier):
    """Simple implementation of Modifier that just runs `fun` on the entire line
       if random() is lower than the modifier's probability.
    """
    fun: Callable[[str], str]

    @classmethod
    def wrap(cls, fn: Callable[[str], str]) -> Type['LineModifier']:
        """Utility function to wrap a function into a Modifier class."""
        return partial(cls, fn)

    def __init__(self, fun: Callable[[str], str], probability:float):
        super().__init__(probability)
        self.fun = fun

    def __call__(self, line:str) -> str:
        return self.fun(line) if self.probability > random.random() else line


@LineModifier.wrap
def TitleCaseModifier(line: str) -> str:
    """Applies titlecase to a sentence. Beware of tabs as src and trg separator
    """
    sections: List[str] = line.split('\t')
    for i in range(len(sections)):
        sections[i] = ' '.join([word[0].upper() + word[1:] for word in sections[i].split()])
    return '\t'.join(sections)


@LineModifier.wrap
def UpperCaseModifier(line: str) -> str:
    return line.upper()
