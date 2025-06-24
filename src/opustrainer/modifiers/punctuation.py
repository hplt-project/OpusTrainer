import random
from typing import List, Iterable

from opustrainer.modifiers import Modifier

# Use regular ASCII punctuation and the corresponding full-width (Chinese) and other analogues
PUNCT_BY_TYPE = {
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
    },
    'ellipsis': {
        '…',  # U+2026  horizontal ellipsis
        '⋯',  # U+22EF  mid-line ellipsis
        '︙',  # U+FE19  vertical CJK ellipsis
        # Don't deal with this case as it complicates handling alignments, skip lines with multiple end punctuation instead
        # '...'  # three consecutive ASCII dots
    }
}

ALL_PUNCT = {mark for _, marks in PUNCT_BY_TYPE.items() for mark in marks}


class RemoveEndPunctuationModifier(Modifier):
    """
    Removes punctuation in the end of the sentence if it matches for the source and target
    """

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
        src, trg = sections[0], sections[1]

        for kind, punct in PUNCT_BY_TYPE.items():
            # Both ends of source and target sentence should have punctuation of the same kind (e.g. `Hello. 你好。`)
            if src[-1] in punct and trg[-1] in punct:
                # Skip lines with multiple terminal punctuation marks (e.g. `Hello!?` or `Hello...`)
                for fragment in [src, trg]:
                    if len(fragment) > 1 and fragment[-2] in ALL_PUNCT:
                        return False
                    # When tokenized (e.g. `Hello . . .`)
                    if len(fragment) > 2 and fragment[-2] == ' ' and fragment[-3] in ALL_PUNCT:
                        return False
                return True

        return False

    def __call__(self, batch: List[str]) -> Iterable[str]:
        for line in batch:
            yield self.remove_punct(line) \
                if self.probability > random.random() and self.ends_with_punct(line) \
                else line
