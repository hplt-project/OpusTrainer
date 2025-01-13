import random
from operator import attrgetter
from pathlib import Path
from typing import Set, List, Tuple, Optional, TypeVar, Iterable

from opustrainer.alignments import Pair, parse_alignments, format_alignments
from opustrainer.modifiers import Modifier
from opustrainer.tokenizers import SpaceDetokenizer, SpaceTokenizer, SentencePieceTokenizer, \
    make_detokenizer, ICU_WHITESPACE_TOKEN
from opustrainer.modifiers.retokenize import Retokenizer, remap_alignment_pairs
from opustrainer import logger


T = TypeVar('T')

def random_weighted_choice(options:Iterable[Tuple[T,float]]) -> T:
    choice = random.random()
    cumsum = 0.0
    for option, prob in options:
        cumsum += prob
        if choice < cumsum:
            return option
    raise RuntimeError('random_weighted_choice called with cumulative sum smaller than 1.0')


def first(iterable:Iterable[T]) -> T:
    """Returns the first value of an iterable. Might raise a StopIteration if iterable is empty"""
    return next(iter(iterable))


def get_random_unicode_words(min_length: int=2, max_length: int=10, max_words: int=3) -> List[str]:
    """Gets a random unicode string of words, of up to max_words, where each word is of length
    min_length-max_length. Only one character set per invocation.
    Maybe should do special rules for emoji and CJK? Emoji wouldn't appear with spaces in between normally
    and CJK would normally be of maximum length of 3 per word."""

    length: int = random.randint(min_length, max_length)

    # Update this to include code point ranges to be sampled
    # https://jrgraphix.net/r/Unicode/
    # https://en.wikipedia.org/wiki/Unicode_block
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
    return [
        ''.join(chr(random.randrange(*alphabet)) for _ in range(length))
        for _ in range(random.randint(1, max_words))
    ]


def filter_one_to_one_pairs(inlist: List[Pair]) -> List[Pair]:
    """Removes places non x->y x->z type of alignments. Remove anything that is found multiple
       times. Anything that is found multiple times means non bijective alignment. Since they
       are sorted, we can reduce some time complexity
    """
    inlist.sort(key=attrgetter('src'))
    new_list: List[Pair] = []
    blacklisted: Set[int] = set()
    for i, pair in enumerate(inlist):
        # Skip blacklisted ones
        if pair.src in blacklisted:
            continue
        
        # Skip (and blacklist) any (x,y) pairs that are followed by (x,_) pairs
        if i < len(inlist) - 1 and pair.src == inlist[i+1].src:
            blacklisted.add(pair.src)
            continue

        # Keep the rest, pairs like (x,_) where `x` did only occur once in the
        # first position among all of the tuples in `inlist`.
        new_list.append(pair)
    return new_list


def get_placeholding_candidates(src_trg: List[Pair]) -> List[Pair]:
    """Filters out multiple alignment targets so that we can definitely get a one-to-one
       replacement
    """
    # Create the two src-trg and trg-src sets
    trg_src: List[Pair] = [Pair(pair.trg, pair.src) for pair in src_trg]

    src_trg_filtered: List[Pair] = filter_one_to_one_pairs(src_trg)
    trg_src_filtered: List[Pair] = filter_one_to_one_pairs(trg_src)

    # Now, we are looking for the union of the two.
    # First, reverse src_trg
    trg_src_filtered_rereversed: List[Pair] = [Pair(pair.trg, pair.src) for pair in trg_src_filtered]

    # Now convert both to sets and take the union
    src_trg_set: Set[Pair] = set(src_trg_filtered)
    trg_src_filtered_rereversed_set: Set[Pair] = set(trg_src_filtered_rereversed)
    selection = src_trg_set & trg_src_filtered_rereversed_set

    # Little dance to make sure we return the original pair instances, not any
    # of the reversed reversed ones. This also sorts them back into the original
    # order.
    return [pair for pair in src_trg if pair in selection]


class PlaceholderTagModifier(Modifier):
    """Unpacks a line, removes the alignments, and applies placeholding. Supports trivial and non trivial detokenization
       using moses detokenizer, which should be used for CJK languages and languages where words are typically not space
       delimited.

       Usage:
       ```yaml
       modifiers:
       - Tags: 0.02
         custom_detok_src: 'moses:zh'
         custom_detok_trg: "moses:null"
         template: "__source__ {src} __target__ {trg} __done__"
         augment: 0.0 # 0% chance to just insert a random string on both sides
         replace: 0.0 # 0% change to use tags to force translate to a random string
        ```
    """

    template: str

    src_retokenizer: Retokenizer
    trg_retokenizer: Retokenizer

    modes: List[Tuple[str,float]]

    # Controls whether alignment info is printed. Normally this is controlled by
    # `spm_vocab` argument, but in tests we also set this to True for testing.
    print_alignments: bool

    def __init__(self, probability: float=0.0, custom_detok_src: Optional[str]=None, custom_detok_trg: Optional[str]=None,
        spm_vocab: Optional[Path]=None,
        template: str="__source__ {src} __target__ {trg} __done__", augment: float=0, replace:float=0, tag:float=1):
        super().__init__(probability)

        self.template = template

        # uses Moses detokenizer by default
        if custom_detok_src and ':' not in custom_detok_src:
            custom_detok_src = f'moses:{custom_detok_src}'
        if custom_detok_trg and ':' not in custom_detok_trg:
            custom_detok_trg = f'moses:{custom_detok_trg}'

        self.custom_detok_src = custom_detok_src
        self.custom_detok_trg = custom_detok_trg

        self.src_retokenizer = Retokenizer(
            detokenizer=make_detokenizer(custom_detok_src) if custom_detok_src else SpaceDetokenizer(),
            tokenizer=SentencePieceTokenizer(spm_vocab) if spm_vocab else SpaceTokenizer()
        )

        self.trg_retokenizer = Retokenizer(
            detokenizer=make_detokenizer(custom_detok_trg) if custom_detok_trg else SpaceDetokenizer(),
            tokenizer=SentencePieceTokenizer(spm_vocab) if spm_vocab else SpaceTokenizer()
        )

        # For now only print alignments when spm_vocab is passed in. We'll improve upon this with #29.
        self.print_alignments = spm_vocab is not None

        self.modes = []

        if augment + replace > 1.0:
            raise ValueError('sum of augment and replace probability should not exceed 1.0')

        if augment > 0:
            self.modes.append(('augment', augment))

        if replace > 0:
            self.modes.append(('replace', replace))

        # the modifier can be used for inline noise augmentation only
        if tag > 0:
            self.modes.append(('tag', tag))

        if ({'replace', 'tag'} & {mode for mode,_ in self.modes}) and \
            'icu' in ((self.custom_detok_trg or '') + (self.custom_detok_trg or '')):
            raise ValueError('ICU tokenization is not supported with "tag" and "replace" modes')

    def __call__(self, batch: List[str]) -> Iterable[str]:
        for line in batch:
            try:
                yield self.apply(line)
            except Exception as exc:
                logger.log(f'Skipping line because of exception: {exc!r}', 'WARNING')

    def apply(self, line:str) -> str:
        """Applies tag to words in a line based on alignment info, and then removes the alignment info from the line.
           This is used to enable terminology support by tagging random words with their translation.
           eg "I like cake" would become "I __source__ like __target__ gusta __done__ cake.
           By default the detokenizer used is the trivial detokenizer, but we can instead have separate detokenizers on src and trg."
        """

        src, trg, *rest = line.strip().split('\t')
        source = src.split()
        target = trg.split()
        alignments = []
        
        # Try parsing alignments. If we fail, the sentence will be thrown out
        # by the trainer.
        alignments = parse_alignments(rest[0], source, target)
        candidate_offset = 0;

        while self.probability > 0.0:
            try:
                # Get replacement candidate. Skip words that are already the same in the source and
                # target sentence. This is to avoid having numbers trained with placeholders or any
                # other words that are exactly the same.
                candidate = first(
                    candidate
                    for candidate in get_placeholding_candidates(alignments[candidate_offset:])
                    if source[candidate.src] != target[candidate.trg]
                )
            except StopIteration:
                # `first` failed because there are no sutable candidates, so lets break out of this
                # loop as well and skip to the output printing part.
                break

            # Candidate pair position in the alignments array
            candidate_index = alignments.index(candidate)

            # Default for if we skip this word: next candidate is found after the current one
            candidate_offset = candidate_index + 1

             # Skip words whose turn it isn't yet.
            if random.random() >= self.probability:
                continue
            
            # Select mode (skip random_weighted_choices*() when 'tag' is the only mode)
            mode = random_weighted_choice(self.modes) if len(self.modes) > 1 else self.modes[0][0]

            if mode == "tag" or mode == "replace":
                if mode == "tag":
                    # Hint the expected output to the trainer by specifying it in the input as a tag
                    augment_tokens = [target[candidate.trg]]
                else:
                    # Same as above, but instead of the expected target word, we replace it on both
                    # sides with a random string. This encourages the model to really ignore the source
                    # and produce the target that we desire.
                    augment_tokens = get_random_unicode_words()
                
                tag_tokens = self.template.format(src=source[candidate.src], trg=' '.join(augment_tokens)).split()
                source = source[:candidate.src] + tag_tokens + source[candidate.src+1:]
                target = target[:candidate.trg] + augment_tokens + target[candidate.trg+1:]

                src_tag_offset = first(n for n, tpl in enumerate(self.template.split()) if '{src}' in tpl)
                trg_tag_offset = first(n for n, tpl in enumerate(self.template.split()) if '{trg}' in tpl)

                # __src__ aaa __trg__ xxx yyy zzz __done__ => xxx yyy zzz
                # ^1      ^2  ^3      ^4  ^5  ^6  ^7          ^1  ^2  ^3
                # So what is the correct alignment pairs for these? For now
                # I map 2-1 2-2 2-3 + 4-1 5-2 6-3?

                # Fix up alignment pairs
                alignments = (
                    # pairs before the replaced bit stay the same
                    alignments[:candidate_index]
                    # fill in the gap created by the replaced bit: the 2-1 2-2 2-3 bit
                    + [Pair(candidate.src + src_tag_offset, candidate.trg + n) for n in range(len(augment_tokens))]
                    # fill in the gap created by the replaced bit: the 4-1 5-2 6-3 bit
                    + [Pair(candidate.src + trg_tag_offset + n, candidate.trg + n) for n in range(len(augment_tokens))]
                    # pairs after the replaced bit have to be offset by the length of the replacement
                    # bit minus the length of the bit we replaced (1)
                    + [Pair(pair.src + len(tag_tokens) - 1, pair.trg) for pair in alignments[candidate_index+1:]]
                )
                candidate_offset = candidate_index + 2 * len(augment_tokens)

            elif mode == "augment":
                # Augment mode adds random noise both on the source and the target without any
                # tagging encouraging the model to copy crap from one side to the other.
                augment_tokens = get_random_unicode_words()
                source, num_src_aug_tokens, pos_aug_src = self.insert_augmented(augment_tokens, source, candidate.src+1, self.custom_detok_src)
                target, num_trg_aug_tokens, pos_aug_trg = self.insert_augmented(augment_tokens, target, candidate.trg+1, self.custom_detok_trg)

                # Fix up alignment pairs
                alignments = (
                    # pairs before and including the candidate stay the same
                    alignments[:candidate_index+1]
                    # fill in the gap created by the added random noise
                    + [Pair(candidate.src + n_src, candidate.trg + n_trg) for n_src, n_trg in zip(pos_aug_src, pos_aug_trg)]
                    # pairs after the replaced bit have to be offset by the length of the replacement bit
                    + [Pair(pair.src + num_src_aug_tokens, pair.trg + num_trg_aug_tokens) for pair in alignments[candidate_index+1:]]
                )
                candidate_offset = candidate_index + min(num_src_aug_tokens, num_trg_aug_tokens) + 1

        source_detok, _, source_mapping = self.src_retokenizer.retokenize(source)
        target_detok, _, target_mapping = self.trg_retokenizer.retokenize(target)
        
        if self.print_alignments:
            remapped_pairs = remap_alignment_pairs(source_mapping, target_mapping, alignments)
            return source_detok + "\t" + target_detok + "\t" + format_alignments(remapped_pairs)
        else:
            return source_detok + "\t" + target_detok

    def insert_augmented(self, augment_tokens: List[str], tokens: List[str], position: int, detokenization: str) -> Tuple[List[str], int, List[int]]:
        """
        Inserts augmented tokens.
        Accounts for possible ICU detokenization which uses special symbol "▁" for whitespace tokens.
            Such tokens will also be inserted to separate the augmented words.

        Returns:
            new tokens
            number of augmented tokens including whitespaces for in ICU case
            alignments positions for the augmented tokens (whitespaces are excluded, we don't need alignments for them)
        """
        prefix = tokens[:position]
        postfix = tokens[position:]
        aug_aln_offset = []

        if detokenization is not None and "icu" in detokenization:
            new_aug_tokens = []
            aug_pos_index = 1

            if len(prefix) > 0 and prefix[-1] != ICU_WHITESPACE_TOKEN:
                new_aug_tokens.append(ICU_WHITESPACE_TOKEN)
                aug_pos_index += 1

            for token in augment_tokens:
                new_aug_tokens.append(token)
                # save the offset of the augmented words to use in alignments
                aug_aln_offset.append(aug_pos_index)
                new_aug_tokens.append(ICU_WHITESPACE_TOKEN)
                aug_pos_index += 2

            if len(postfix) > 0 and postfix[0] == ICU_WHITESPACE_TOKEN:
                new_aug_tokens.pop()

            augment_tokens = new_aug_tokens
        else:
            aug_aln_offset = list(range(1, len(augment_tokens) + 1))

        tokens = prefix + augment_tokens + postfix
        return tokens, len(augment_tokens), aug_aln_offset

    def validate(self, context:List[Modifier]) -> None:
        """Current limitation of the tags modifier is that any other modifier might modify the
        inserted tags, which we don't want. So warn users about that if we notice it.
        """
        if context[-1] != self:
            logger.log('Tags modifier should to be the last modifier to be applied, as otherwise other modifiers might alter the inserted tags themselves.', loglevel="WARNING")
