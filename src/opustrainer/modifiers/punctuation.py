import random
from typing import List, Iterable

from opustrainer.modifiers import Modifier


class RemoveEndPunctuationModifier(Modifier):
    """
    Removes punctuation in the end of the sentence if it matches for the source and target
    """

    # Use regular ASCII punctuation and the corresponding full-width (Chinese) and other analogues
    PUNCT = {
        # PERIOD / FULL STOP
        'period': {
            '.', '．', '｡', '。',  # ASCII + CJK full/half-width
            '﹒', '︒',  # CJK small / presentation
            '।', '॥',  # danda, double-danda  (hi, bn, gu, kn, ml, mr, ta, te)
            '۔',  # Arabic full stop     (ar, fa, ur)
            '။',  # Myanmar stop         (my)
            '።',  # Ethiopic stop        (am)
            '։'  # Armenian stop        (hy)
        },

        # EXCLAMATION MARK
        'exclamation': {
            '!', '！',  # ASCII + CJK full-width
            '﹗', '︕',  # CJK small / presentation
            '‼', '❢', '❣',  # double & heavy
            '՜'  # Armenian exclamation (hy)
        },

        # QUESTION MARK
        'question': {
            '?', '？',  # ASCII + CJK full-width
            '﹖', '︖',  # CJK small / presentation
            '؟',  # Arabic question      (ar, fa, ur)
            ';',  # Greek question mark  (el)
            '՞',  # Armenian question    (hy)
            '፧',  # Ethiopic question    (am)
            '⁇', '⁈', '⁉'  # double / interrobang
        },

        # COMMA
        'comma': {
            ',', '，', '､',  # ASCII + CJK full/half-width
            '﹐', '︐',  # CJK small / presentation
            '、', '﹑', '︑',  # ideographic comma variants
            '،',  # Arabic comma         (ar, fa, ur)
            '၊',  # Myanmar comma        (my)
            '՝'  # Armenian comma       (hy)
        },

        # COLON
        'colon': {
            ':', '：',  # ASCII + CJK full-width
            '﹕', '︓',  # CJK small / presentation
            '፥', '፦',  # Ethiopic colons       (am)
            '·'  # Greek ano-teleia — semicolon/colon (el)
        },

        # SEMICOLON
        'semicolon': {
            ';', '；',  # ASCII + CJK full-width
            '﹔', '︔',  # CJK small / presentation
            '؛',  # Arabic semicolon     (ar, fa, ur)
            '፤'  # Ethiopic semicolon   (am)
        }
    }

    def remove_punct(self, line: str) -> str:
        sections: List[str] = line.split('\t')
        src = sections[0]
        trg = sections[1]
        out_sections = [src[:-1].rstrip(), trg[:-1].rstrip()]

        if len(sections) == 3:
            aln = sections[2]
            puct_pos_src = len(src.split(' ')) - 1
            puct_pos_trg = len(trg.split(' ')) - 1
            # alignments format: 0-0 1-1 2-4 5-6
            # remove pairs that include end punctuation positions
            new_aln_parts = [pair for pair in aln.split(' ') if
                             int(pair.split('-')[0]) != puct_pos_src and int(pair.split('-')[1]) != puct_pos_trg]
            new_aln = ' '.join(new_aln_parts)
            out_sections.append(new_aln)

        return '\t'.join(out_sections)

    def ends_with_punct(self, line: str) -> bool:
        sections: List[str] = line.split('\t')

        for kind, punct in self.PUNCT.items():
            # Both ends of source and target sentence should have punctuation of the same kind
            if sections[0][-1] in punct and sections[1][-1] in punct:
                return True

        return False

    def __call__(self, batch: List[str]) -> Iterable[str]:
        for line in batch:
            yield self.remove_punct(line) \
                if self.probability > random.random() and self.ends_with_punct(line) \
                else line
