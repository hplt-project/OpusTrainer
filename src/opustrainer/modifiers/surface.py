import random
from typing import List, Iterable

from opustrainer.modifiers import Modifier


class TitleCaseModifier(Modifier):
    """Applies titlecase to a sentence. Beware of tabs as src and trg separator
    """
    def __call__(self, batch:List[str]) -> Iterable[str]:
        for line in batch:
            if self.probability <= random.random():
                yield line
            else:
                sections: List[str] = line.split('\t')
                for i in range(len(sections)):
                    sections[i] = ' '.join([word[0].upper() + word[1:] for word in sections[i].split()])
                yield '\t'.join(sections)


class UpperCaseModifier(Modifier):
    def __call__(self, batch:List[str]) -> Iterable[str]:
        for line in batch:
            yield line.upper() if self.probability > random.random() else line
