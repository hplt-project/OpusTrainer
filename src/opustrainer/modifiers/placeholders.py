import random
from operator import itemgetter
from typing import Set, List, Tuple, Optional, Protocol

from sacremoses import MosesDetokenizer

from opustrainer.modifiers import Modifier


def tuplify(pair: str) -> Tuple[int, int]:
    """Parses "x-y" description of an aligned token pair into `(x,y)` tuple of ints."""
    s, t = pair.split('-')
    return (int(s), int(t))


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


def get_placeholding_candidates(align_line: str) -> List[Tuple[int, int]]:
    """Filters out multiple alignment targets so that we can definitely get a one-to-one
       replacement
    """
    # Create the two src-trg and trg-src sets
    src_trg: List[Tuple[int, int]] = [tuplify(i) for i in align_line.split()]
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
    """Unpacks a line, removes the alignments, and applies placeholding. Hardcoded for the moment.
       Also applies detokenization on the source side, because getting word alignments for Chinese
       is otherwise hard.
    """

    num_tags: int
    template: str
    src_detokenizer: Detokenizer
    trg_detokenizer: Optional[Detokenizer]

    def __init__(self, probability: float=0.0, num_tags: int=6,
        custom_detok_src: Optional[str]=None, custom_detok_trg: Optional[str] = None,
        template=" <tag{0:d}> {1} </tag{0:d}>"):
        super().__init__(probability)

        self.num_tags = num_tags
        self.template = template

        if custom_detok_src:
            self.src_detokenizer = MosesDetokenizer(lang=custom_detok_src)
        else:
            self.src_detokenizer = SpaceDetokenizer()

        if custom_detok_trg:
            self.trg_detokenizer = MosesDetokenizer(lang=custom_detok_trg)
        else:
            self.trg_detokenizer = None


    def __call__(self, line:str) -> str:
        """Applies tag to words in a line based on alignment info, and then removes the alignment info from the line.
           This is used to enable terminology support by tagging random words with their translation.
           eg "I like cake" would become "I like <tag0> gusta </tag0> cake. By default the detokenizer used is the trivial
           detokenizer, but we can instead have separate detokenizers on src and trg."
        """
        src, trg, alignment = line.strip().split('\t')
        source = src.split(' ')
        target = trg.split(' ')

        # Get replacement candidates
        candidates: List[Tuple[int, int]] = get_placeholding_candidates(alignment)

        # Get list of possible tags.
        tags: List[int] = list(range(self.num_tags))
        # Shuffle the list. It's quite slow, but we only have 6 elements so it should be fine
        # For more information https://stackoverflow.com/questions/10048069/what-is-the-most-pythonic-way-to-pop-a-random-element-from-a-list
        random.shuffle(tags)

        # Replace each of them with a THRESHOLD probability unless the two words are exactly the same
        # This is to avoid having numbers trained with placeholders or any other words that are exactly the same
        for i in range(len(candidates)):
            # Skip words that are the same on 
            if source[candidates[i][0]] == target[candidates[i][1]]:
                continue
            
            # Skip words whose turn it isn't yet.
            if random.random() >= self.probability:
                continue

            # We run out of tags so no point of trying to tag anything else
            if not tags:
                break

            tag_id = tags.pop()
            source[candidates[i][0]] = source[candidates[i][0]] + self.template.format(tag_id, target[candidates[i][1]])

        source_detok: str = self.src_detokenizer.detokenize(source)
        if self.trg_detokenizer is not None:
            trg = self.trg_detokenizer.detokenize(target)

        # Return the sentence, source tagged a la Dinu et al, target as it is and no alignment info
        return  source_detok + "\t" + trg
