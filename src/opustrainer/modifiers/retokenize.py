from typing import List, Protocol, Dict, NamedTuple, TypeVar, Callable, Union, Tuple, Optional, Any
from itertools import count
import sacremoses
from sentencepiece import SentencePieceProcessor

from opustrainer.modifiers import Modifier
from opustrainer import logger


TokenList = List[str] # todo: bytes?

TokenMapping = List[List[int]]


class Detokenizer(Protocol):
  """Turns a list of tokens into a string"""
  def detokenize(self, tokens:TokenList) -> Tuple[str, List[slice]]:
    ...


class Tokenizer(Protocol):
  def tokenize(self, text:str) -> Tuple[TokenList,List[slice]]:
    ...


class SpaceDetokenizer:
  def detokenize(self, tokens:TokenList) -> Tuple[str,List[slice]]:
    spans = []
    offset = 0
    for token in tokens:
      spans.append(slice(offset, offset + len(token)))
      offset += len(token) + 1 # space
    return ' '.join(tokens), spans


class MosesDetokenizer:
  def __init__(self, lang:str):
    self.detokenizer = sacremoses.MosesDetokenizer(lang)

  def detokenize(self, tokens:TokenList) -> Tuple[str,List[slice]]:
    text = self.detokenizer.detokenize(tokens)
    spans = []
    offset = 0
    for token in tokens:
      offset = text.find(token, offset)
      if offset == -1:
        raise RuntimeError(f"Could not find token '{token}' in detokenized text")
      spans.append(slice(offset, offset + len(token)))
      offset += len(token)
    return text, spans


class SentencePieceTokenizer:
  def __init__(self, vocab:str):
    self.spm = SentencePieceProcessor(vocab)

  def tokenize(self, text:str) -> Tuple[TokenList,List[slice]]:
    # interestingly, piece.begin and piece.end are unicode offsets, not byte
    # offsets as the documentation would suggest. When byte-fallback happens,
    # there will be pieces where piece.begin and piece.end are the same value
    # but they are technically necessary to encode the following pieces.
    # e.g:
    # > x.encode('🤣', out_type='immutable_proto').pieces
    #   { piece: "▁" id: 275 surface: "" begin: 0 end: 0 }
    #   { piece: "<0xF0>" id: 247 surface: "" begin: 0 end: 0 }
    #   { piece: "<0x9F>" id: 166 surface: "" begin: 0 end: 0 }
    #   { piece: "<0xA4>" id: 171 surface: "" begin: 0 end: 0 }
    #   { piece: "<0xA3>" id: 170 surface: "🤣" begin: 0 end: 1 }
    # > x.decode([247,166,171,170])
    #   '🤣'
    spans = [
      slice(piece.begin, piece.end)
      for piece in self.spm.encode(text.encode(), out_type='immutable_proto').pieces
    ]
    tokens = [text[span] for span in spans]
    return tokens, spans

def overlaps(r1:slice, r2:slice) -> bool:
  # (a,b), (x,y) = r1, r2
  #      [a    b]             | a < y |  x < b
  # [x y]                 = F |   F   |    T
  #     [x y]             = T |   T   |    T
  #         [x y]         = T |   T   |    T
  #            [x y]      = T |   T   |    T
  #                [x  y] = F |   T   |    F
  return r1.start < r2.stop and r2.start < r1.stop


def print_cells(*rows, format:Callable[[Any],str]=repr,file=None) -> None:
  # Convert it all to text
  text_rows = [
    [format(cell) for cell in row]
    for row in rows
  ]
  
  # Calculate column width
  widths = [
    max(len(cell) for cell in column)
    for column in zip(*text_rows)
  ]

  # Print rows
  for row in text_rows:
    for width, cell in zip(widths, row):
      print(f'{{:<{width:d}}} '.format(cell), end='', file=file)
    print(file=file)


def short_repr(obj):
  if isinstance(obj, slice):
    return f'({obj.start},{obj.stop})'
  else:
    return repr(obj)


class Retokenizer(NamedTuple):
  detokenizer: Detokenizer
  tokenizer: Tokenizer

  def retokenize(self, tokens:TokenList) -> Tuple[str,TokenList,TokenMapping]:
    detokenized, old_token_spans = self.detokenizer.detokenize(tokens)
    new_tokens, new_token_spans = self.tokenizer.tokenize(detokenized)

    old_to_new_mapping = [[] for _ in range(len(old_token_spans))]

    # print(f"{detokenized=}")
    # print_cells(new_tokens, new_token_spans, format=short_repr)
    # print(f"{new_token_spans=}")

    #TODO: This can be done much more efficiently
    for i, old_token_span in enumerate(old_token_spans):
      for j, new_token_span in enumerate(new_token_spans):
        if overlaps(old_token_span, new_token_span):
          old_to_new_mapping[i].append(j)

    # for n, old_token_span, new_token_indices in zip(count(), old_token_spans, old_to_new_mapping):
    #   print(f'<{n}>[{detokenized[old_token_span]}]({old_token_span.start},{old_token_span.stop}): ' + ' '.join(
    #     f'<{new_idx}>[{detokenized[new_token_spans[new_idx]]}]({new_token_spans[new_idx].start},{new_token_spans[new_idx].stop})'
    #     for new_idx in new_token_indices
    #   ))
    # print()

    return detokenized, new_tokens, old_to_new_mapping


T = TypeVar('T', bound=Union[Tokenizer,Detokenizer])

def make_tokenizer(implementations: Dict[str,Callable[...,T]], spec:str) -> T:
  name, *args = spec.split(':')
  return implementations[name](*args)


def make_retokenizer(spec:Dict[str,str]) -> Retokenizer:
  return Retokenizer(
    detokenizer=make_tokenizer(DETOKENIZERS, spec.get('detokenize', 'spaces')),
    tokenizer=make_tokenizer(TOKENIZERS, spec.get('tokenize', 'spaces'))
  )


class Pair(NamedTuple):
  src:int
  trg:int


def parse_alignments(pairs:str, src_tokens:Optional[TokenList]=None, trg_tokens:Optional[TokenList]=None) -> List[Pair]:
  pairs = [
    Pair(int(a), int(b)) for a, b in
    (pair.split('-', maxsplit=1) for pair in pairs.split())
  ]

  if src_tokens is not None and trg_tokens is not None:
    invalid_pairs = [
      pair
      for pair in pairs
      if pair.src < 0 or pair.src >= len(src_tokens)
      or pair.trg < 0 or pair.trg >= len(trg_tokens)
    ]
    if invalid_pairs:
      raise ValueError('Out-of-bound alignment pairs: ' + ' '.join(map(repr, invalid_pairs)))

  return pairs


def format_alignments(pairs:List[Pair]) -> str:
  return ' '.join(f'{pair.src}-{pair.trg}' for pair in pairs)


def compute_mapping(src_mapping:TokenMapping, trg_mapping:TokenMapping, alignments:List[Pair]) -> List[Pair]:
  remapped = set()
  for old_src_idx, old_trg_idx in alignments:
    for src_idx in src_mapping[old_src_idx]:
      for trg_idx in trg_mapping[old_trg_idx]:
        remapped.add(Pair(src_idx, trg_idx))
  return sorted(remapped)


DETOKENIZERS = {
  'moses': lambda lang: MosesDetokenizer(lang),
  'spaces': lambda: SpaceDetokenizer(),
}

TOKENIZERS = {
  'moses': lambda lang: MosesTokenizer(lang),
  'spm': lambda vocab: SentencePieceTokenizer(vocab),
  'spaces': lambda: SpaceTokenizer(),
}


class RetokenizeModifier(Modifier):
  src: Retokenizer
  trg: Retokenizer

  def __init__(self, probability: float=0.0, src:dict=dict(), trg:dict=dict()):
    super().__init__(probability) # probability is very much ignored lol.
    self.src = make_retokenizer(src)
    self.trg = make_retokenizer(trg)

  def __call__(self, line:str) -> str:
    src, trg, alignments = line.split('\t')
    src_tokens = src.split()
    trg_tokens = trg.split()
    pairs = parse_alignments(alignments, src_tokens, trg_tokens)
    new_src, new_src_tokens, src_mapping = self.src.retokenize(src_tokens)
    new_trg, new_trg_tokens, trg_mapping = self.trg.retokenize(trg_tokens)
    remapped_pairs = compute_mapping(src_mapping, trg_mapping, pairs)
    return '\t'.join((new_src, new_trg, format_alignments(remapped_pairs)))

