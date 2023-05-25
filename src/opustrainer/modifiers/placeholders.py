import random
from operator import itemgetter
from typing import Set, List, Tuple, Optional, Protocol, TypeVar, Iterable
from warnings import warn

from sacremoses import MosesDetokenizer

from opustrainer.modifiers import Modifier


T = TypeVar('T')

def random_weighted_choice(options:Iterable[Tuple[T,float]]) -> T:
    choice = random.random()
    cumsum = 0
    for option, prob in options:
        cumsum += prob
        if choice < cumsum:
            return option
    raise RuntimeError('random_weighted_choice called with cumulative sum smaller than 1.0')



def get_random_unicode_string(min_length: int=1, max_length: int=4, max_words: int=3) -> str:
    """Gets a random unicode string of words, of up to max_words, where each word is of length
    min_length-max_length. Only one character set per invocation.
    Maybe should do special rules for emoji and CJK? Emoji wouldn't appear with spaces in between normally
    and CJK would normally be of maximum length of 3 per word."""

    length: int = random.randint(min_length, max_length)

    # Update this to include code point ranges to be sampled
    # https://jrgraphix.net/r/Unicode/
    # https://en.wikipedia.org/wiki/Unicode_block
    # Mixing left to righ and right to left bytes causes issues. A lot of them.
    include_ranges = [
        (0x0020, 0x007F), # Basic Latin
        (0x00A0, 0x00FF), # Latin-1 Supplement
       # (0x0100, 0x017F), # Latin Extended-A
       # (0x0180, 0x024F), # Latin Extended-B
       # (0x0250, 0x02AF), # IPA Extensions
       # (0x02B0, 0x02FF), # Spacing Modifier Letters
       # (0x0300, 0x036F), # Combining Diacritical Marks
        (0x0370, 0x03FF), # Greek and Coptic
        (0x0400, 0x04FF), # Cyrillic
       # (0x0500, 0x052F), # Cyrillic Supplementary
        (0x0530, 0x058F), # Armenian
        (0x0590, 0x05FF), # Hebrew
        (0x0600, 0x06FF), # Arabic
       # (0x0700, 0x074F), # Syriac
       # (0x0780, 0x07BF), # Thaana
        (0x0900, 0x097F), # Devanagari
        (0x0980, 0x09FF), # Bengali
       # (0x0A00, 0x0A7F), # Gurmukhi
        (0x0A80, 0x0AFF), # Gujarati
       # (0x0B00, 0x0B7F), # Oriya
       # (0x0B80, 0x0BFF), # Tamil
       # (0x0C00, 0x0C7F), # Telugu
       # (0x0C80, 0x0CFF), # Kannada
       # (0x0D00, 0x0D7F), # Malayalam
       # (0x0D80, 0x0DFF), # Sinhala
        (0x0E00, 0x0E7F), # Thai
       # (0x0E80, 0x0EFF), # Lao
       # (0x0F00, 0x0FFF), # Tibetan
        (0x1000, 0x109F), # Myanmar
        (0x10A0, 0x10FF), # Georgian
       # (0x1100, 0x11FF), # Hangul Jamo
       # (0x1200, 0x137F), # Ethiopic
       # (0x13A0, 0x13FF), # Cherokee
       # (0x1400, 0x167F), # Unified Canadian Aboriginal Syllabics
       # (0x1680, 0x169F), # Ogham
       # (0x16A0, 0x16FF), # Runic
       # (0x1700, 0x171F), # Tagalog
       # (0x1720, 0x173F), # Hanunoo
       # (0x1740, 0x175F), # Buhid
       # (0x1760, 0x177F), # Tagbanwa
        (0x1780, 0x17FF), # Khmer
       # (0x1800, 0x18AF), # Mongolian
       # (0x1900, 0x194F), # Limbu
       # (0x1950, 0x197F), # Tai Le
       # (0x19E0, 0x19FF), # Khmer Symbols
       # (0x1D00, 0x1D7F), # Phonetic Extensions
       # (0x1E00, 0x1EFF), # Latin Extended Additional
       # (0x1F00, 0x1FFF), # Greek Extended
       # (0x2000, 0x206F), # General Punctuation
       # (0x2070, 0x209F), # Superscripts and Subscripts
       # (0x20A0, 0x20CF), # Currency Symbols
       # (0x20D0, 0x20FF), # Combining Diacritical Marks for Symbols
       # (0x2100, 0x214F), # Letterlike Symbols
       # (0x2150, 0x218F), # Number Forms
       # (0x2190, 0x21FF), # Arrows
       # (0x2200, 0x22FF), # Mathematical Operators
       # (0x2300, 0x23FF), # Miscellaneous Technical
       # (0x2400, 0x243F), # Control Pictures
       # (0x2440, 0x245F), # Optical Character Recognition
       # (0x2460, 0x24FF), # Enclosed Alphanumerics
       # (0x2500, 0x257F), # Box Drawing
       # (0x2580, 0x259F), # Block Elements
       # (0x25A0, 0x25FF), # Geometric Shapes
       # (0x2600, 0x26FF), # Miscellaneous Symbols
       # (0x2700, 0x27BF), # Dingbats
       # (0x27C0, 0x27EF), # Miscellaneous Mathematical Symbols-A
       # (0x27F0, 0x27FF), # Supplemental Arrows-A
       # (0x2800, 0x28FF), # Braille Patterns
       # (0x2900, 0x297F), # Supplemental Arrows-B
       # (0x2980, 0x29FF), # Miscellaneous Mathematical Symbols-B
       # (0x2A00, 0x2AFF), # Supplemental Mathematical Operators
       # (0x2B00, 0x2BFF), # Miscellaneous Symbols and Arrows
       # (0x2E80, 0x2EFF), # CJK Radicals Supplement
       # (0x2F00, 0x2FDF), # Kangxi Radicals
       # (0x2FF0, 0x2FFF), # Ideographic Description Characters
       # (0x3000, 0x303F), # CJK Symbols and Punctuation
        (0x3040, 0x309F), # Hiragana
        (0x30A0, 0x30FF), # Katakana
       # (0x3100, 0x312F), # Bopomofo
       # (0x3130, 0x318F), # Hangul Compatibility Jamo
       # (0x3190, 0x319F), # Kanbun
       # (0x31A0, 0x31BF), # Bopomofo Extended
       # (0x31F0, 0x31FF), # Katakana Phonetic Extensions
       # (0x3200, 0x32FF), # Enclosed CJK Letters and Months
       # (0x3300, 0x33FF), # CJK Compatibility
       # (0x3400, 0x4DBF), # CJK Unified Ideographs Extension A
       # (0x4DC0, 0x4DFF), # Yijing Hexagram Symbols
        (0x4E00, 0x9FFF), # CJK Unified Ideographs
       # (0xA000, 0xA48F), # Yi Syllables
       # (0xA490, 0xA4CF), # Yi Radicals
        (0xAC00, 0xD7AF), # Hangul Syllables
       # (0xD800, 0xDB7F), # High Surrogates
       # (0xDB80, 0xDBFF), # High Private Use Surrogates
       # (0xDC00, 0xDFFF), # Low Surrogates
       # (0xE000, 0xF8FF), # Private Use Area
       # (0xF900, 0xFAFF), # CJK Compatibility Ideographs
       # (0xFB00, 0xFB4F), # Alphabetic Presentation Forms
       # (0xFB50, 0xFDFF), # Arabic Presentation Forms-A
       # (0xFE00, 0xFE0F), # Variation Selectors
       # (0xFE20, 0xFE2F), # Combining Half Marks
       # (0xFE30, 0xFE4F), # CJK Compatibility Forms
       # (0xFE50, 0xFE6F), # Small Form Variants
       # (0xFE70, 0xFEFF), # Arabic Presentation Forms-B
       # (0xFF00, 0xFFEF), # Halfwidth and Fullwidth Forms
       # (0xFFF0, 0xFFFF), # Specials
       # (0x10000, 0x1007F), # Linear B Syllabary
       # (0x10080, 0x100FF), # Linear B Ideograms
       # (0x10100, 0x1013F), # Aegean Numbers
       # (0x10300, 0x1032F), # Old Italic
       # (0x10330, 0x1034F), # Gothic
       # (0x10380, 0x1039F), # Ugaritic
       # (0x10400, 0x1044F), # Deseret
       # (0x10450, 0x1047F), # Shavian
       # (0x10480, 0x104AF), # Osmanya
       # (0x10800, 0x1083F), # Cypriot Syllabary
       # (0x1D000, 0x1D0FF), # Byzantine Musical Symbols
       # (0x1D100, 0x1D1FF), # Musical Symbols
       # (0x1D300, 0x1D35F), # Tai Xuan Jing Symbols
       # (0x1D400, 0x1D7FF), # Mathematical Alphanumeric Symbols
        (0x1F600, 0x1F64F),  # Emoji
       # (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
       # (0x2F800, 0x2FA1F), # CJK Compatibility Ideographs Supplement
       # (0xE0000, 0xE007F), # Tags
    ]
    # Select a character set
    alphabet = random.choice(include_ranges)
    
    # Generate a random string of 1 - 3 words
    return ' '.join(
        ''.join(chr(random.randrange(*alphabet)) for _ in range(length))
        for _ in range(random.randint(1, max_words))
    )


def filter_tuples(inlist: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Removes places non x->y x->z type of alignments. Remove anything that is found multiple
       times. Anything that is found multiple times means non bijective alignment. Since they
       are sorted, we can reduce some time complexity
    """
    inlist.sort(key=itemgetter(0))
    new_list: List[Tuple[int, int]] = []
    blacklisted: Set[int] = set()
    for i, (curr, _) in enumerate(inlist):
        # Skip blacklisted ones
        if curr in blacklisted:
            continue
        
        # Skip (and blacklist) any (x,y) pairs that are followed by (x,_) pairs
        if i < len(inlist) - 1 and curr == inlist[i+1][0]:
            blacklisted.add(curr)
            continue

        # Keep the rest, pairs like (x,_) where `x` did only occur once in the
        # first position among all of the tuples in `inlist`.
        new_list.append(inlist[i])
    return new_list


def parse_alignments(line:str) -> List[Tuple[int,int]]:
    """Parses list of "x-y" description of an aligned token pair into `(x,y)` tuples of ints."""
    return [(int(s), int(t)) for s, t in (pair.split('-') for pair in line.split())]


def get_placeholding_candidates(alignments: List[Tuple[int,int]]) -> List[Tuple[int, int]]:
    """Filters out multiple alignment targets so that we can definitely get a one-to-one
       replacement
    """
    # Create the two src-trg and trg-src sets
    src_trg: List[Tuple[int, int]] = alignments
    trg_src: List[Tuple[int, int]] = [(c, d) for d,c in src_trg]

    src_trg_filtered: List[Tuple[int, int]] = filter_tuples(src_trg)
    trg_src_filtered: List[Tuple[int, int]] = filter_tuples(trg_src)

    # Now, we are looking for the union of the two.
    # First, reverse src_trg
    trg_src_filtered_rereversed: List[Tuple[int, int]] = [(c, d) for d,c in trg_src_filtered]

    # Now convert both to sets and take the union
    src_trg_set: Set[Tuple[int, int]] = set(src_trg_filtered)
    trg_src_filtered_rereversed_set: Set[Tuple[int, int]] = set(trg_src_filtered_rereversed)

    return list(src_trg_set & trg_src_filtered_rereversed_set)


class Detokenizer(Protocol):
    def detokenize(self, tokens:List[str]) -> str:
        ...


class SpaceDetokenizer:
    def detokenize(self, tokens:List[str]) -> str:
        return ' '.join(tokens)


class PlaceholderTagModifier(Modifier):
    """Unpacks a line, removes the alignments, and applies placeholding. Supports trivial and non trivial detokenization
       using moses detokenizer, which should be used for CJK languages and languages where words are typically not space
       delimited.

       Usage:
       ```yaml
       modifiers:
       - Tags: 0.02
         custom_detok_src: 'zh'
         custom_detok_trg: null
         template: "__source__ {src} __target__ {trg} __done__"
         augment: 0.0 # 0% chance to just insert a random string on both sides
         replace: 0.0 # 0% change to use tags to force translate to a random string
        ```
    """

    template: str
    src_detokenizer: Detokenizer
    trg_detokenizer: Detokenizer
    modes: List[Tuple[str,float]]

    def __init__(self, probability: float=0.0, custom_detok_src: Optional[str]=None, custom_detok_trg: Optional[str]=None,
        template: str="__source__ {src} __target__ {trg} __done__", augment: float=0, replace:float=0):
        super().__init__(probability)

        self.template = template

        if custom_detok_src:
            self.src_detokenizer = MosesDetokenizer(lang=custom_detok_src)
        else:
            self.src_detokenizer = SpaceDetokenizer()

        if custom_detok_trg:
            self.trg_detokenizer = MosesDetokenizer(lang=custom_detok_trg)
        else:
            self.trg_detokenizer = SpaceDetokenizer()

        self.modes = []

        if augment + replace > 1.0:
            raise ValueError('sum of augment and replace probability should not exceed 1.0')

        if augment > 0:
            self.modes.append(('augment', augment))

        if replace > 0:
            self.modes.append(('replace', replace))

        self.modes.append(('tag', 1.0)) # Weight doesn't matter as long as cumsum => 1.0, it's last on the list anyway

    def __call__(self, line:str) -> str:
        """Applies tag to words in a line based on alignment info, and then removes the alignment info from the line.
           This is used to enable terminology support by tagging random words with their translation.
           eg "I like cake" would become "I __source__ like __target__ gusta __done__ cake. 
           By default the detokenizer used is the trivial detokenizer, but we can instead have separate detokenizers on src and trg."
        """

        src, trg, alignment = line.strip().split('\t')
        source = src.split(' ')
        target = trg.split(' ')
        alignments = parse_alignments(alignment)

        if max(s for s, _ in alignments) >= len(source):
            raise ValueError('alignment pair source token index out of range')
        if max(t for _, t in alignments) >= len(target):
            raise ValueError('alignment pair target token index out of range')

        # Get replacement candidates
        candidates: List[Tuple[int, int]] = get_placeholding_candidates(alignments)

        # Replace each of them with a THRESHOLD probability unless the two words are exactly the same
        # This is to avoid having numbers trained with placeholders or any other words that are exactly the same
        for i in range(len(candidates)):
            # Skip words that are the same on 
            if source[candidates[i][0]] == target[candidates[i][1]]:
                continue

            # Skip words whose turn it isn't yet.
            if random.random() >= self.probability:
                continue
            
            # Select mode (skip random_weighted_choices*() when 'tag' is the only mode)
            mode = random_weighted_choice(self.modes) if len(self.modes) > 1 else 'tag'

            if mode == "tag":
                # Hint the expected output to the trainer by specifying it in the input as a tag
                source[candidates[i][0]] = self.template.format(src=source[candidates[i][0]], trg=target[candidates[i][1]])
            elif mode == "replace":
                # Same as above, but instead of the expected target word, we replace it on both
                # sides with a random string. This encourages the model to really ignore the source
                # and produce the target that we desire.
                augment = get_random_unicode_string()
                source[candidates[i][0]] = self.template.format(src=source[candidates[i][0]], trg=augment)
                target[candidates[i][1]] = augment
            elif mode == "augment":
                # Augment mode adds random noise both on the source and the target without any
                # tagging encouraging the model to copy crap from one side to the other.
                augment = get_random_unicode_string()
                source[candidates[i][0]] = source[candidates[i][0]] + " " + augment
                target[candidates[i][1]] = target[candidates[i][1]] + " " + augment

        source_detok = self.src_detokenizer.detokenize(source)
        target_detok = self.trg_detokenizer.detokenize(target)

        # Return the sentence, source tagged a la Dinu et al, target as it is and no alignment info
        return  source_detok + "\t" + target_detok

    def validate(self, context:List[Modifier]) -> None:
        """Current limitation of the tags modifier is that any other modifier might modify the
        inserted tags, which we don't want. So warn users about that if we notice it.
        """
        if context[-1] != self:
            warn('Tags modifier should to be the last modifier to be applied, as otherwise other modifiers might alter the inserted tags themselves.')
