import re
from typing import Tuple, List, TypeVar, Callable, Union, Dict, Optional
from pathlib import Path

import sacremoses
from sentencepiece import SentencePieceProcessor

from opustrainer.types import TokenList, TokenSpanList, Tokenizer, Detokenizer


DETOKENIZERS = {
    'moses': lambda lang: MosesDetokenizer(lang),
    'spaces': lambda: SpaceDetokenizer(),
    'icu': lambda lang: IcuDetokenizer(lang),

}

TOKENIZERS = {
    'moses': lambda lang: MosesTokenizer(lang),
    'spm': lambda vocab: SentencePieceTokenizer(vocab),
    'spaces': lambda: SpaceTokenizer(),
    'icu': lambda lang: IcuTokenizer(lang),
}


T = TypeVar('T', bound=Union[Tokenizer,Detokenizer])

def _make(implementations: Dict[str,Callable[...,T]], spec:str) -> T:
    name, *args = spec.split(':')
    return implementations[name](*args)

def make_detokenizer(spec:str) -> Detokenizer:
    """Creates a Detokenizer using a spec, e.g. `moses:en` or spm:path/to/vocab.spm`."""
    return _make(DETOKENIZERS, spec)

def make_tokenizer(spec:str) -> Tokenizer:
    """Creates a Tokenizer using a spec, e.g. `moses:en` or spm:path/to/vocab.spm`."""
    return _make(TOKENIZERS, spec)


class SpaceTokenizer:
    """Splits 'Hello World.' into `[Hello, World.]`."""
    def tokenize(self, text:str) -> Tuple[TokenList, TokenSpanList]:
        tokens: TokenList = []
        spans: TokenSpanList = []
        for match in re.finditer(r'[^\s]+', text):
            tokens.append(match.group(0))
            spans.append(slice(match.start(0), match.end(0)))
        return tokens, spans


class SpaceDetokenizer:
    """Turns `[Hello, World.]` back into `Hello World.`."""
    def detokenize(self, tokens:TokenList) -> Tuple[str,TokenSpanList]:
        spans = []
        offset = 0
        for token in tokens:
            spans.append(slice(offset, offset + len(token)))
            offset += len(token) + 1 # space
        return ' '.join(tokens), spans


class MosesTokenizer:
    """Turns `Hello World.` into `[Hello, World, .]` according to Moses and,
    if available, language specific rules."""
    tokenizer: sacremoses.MosesTokenizer

    def __init__(self, lang:str, custom_nonbreaking_prefixes:Optional[str]=None):
        self.tokenizer = sacremoses.MosesTokenizer(lang, custom_nonbreaking_prefixes)

    def tokenize(self, text:str) -> Tuple[TokenList, TokenSpanList]:
        tokens:TokenList = self.tokenizer.tokenize(text, escape=False) # type: ignore
        spans: TokenSpanList = [] # ^tokenizer.tokenize always returns a string unless return_string=True (which is not)
        offset = 0
        for token in tokens:
            offset = text.find(token, offset)
            if offset == -1:
                raise RuntimeError(f"Could not find token '{token}' in original text")
            spans.append(slice(offset, offset + len(token)))
            offset += len(token)
        return tokens, spans


class MosesDetokenizer:
    """Turns `[Hello,World,.]` back into `Hello World.`. Rules can be language-specific."""
    detokenizer: sacremoses.MosesDetokenizer

    def __init__(self, lang:str):
        self.detokenizer = sacremoses.MosesDetokenizer(lang)

    def detokenize(self, tokens:TokenList) -> Tuple[str,TokenSpanList]:
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
    """Turns `Hello World.` into something like [He,llo,_World,.] depending on your vocab."""
    spm: SentencePieceProcessor

    def __init__(self, vocab:Path):
        self.spm = SentencePieceProcessor(model_file=str(vocab)) # type: ignore # for some reason pylance doesn't understand spm.

    def tokenize(self, text:str) -> Tuple[TokenList,TokenSpanList]:
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
            for piece in self.spm.encode(text.encode(), out_type='immutable_proto').pieces # type: ignore
        ]
        tokens = [text[span] for span in spans]
        return tokens, spans

# The same character as in SentencePiece
ICU_WHITESPACE_TOKEN = "▁"
class IcuTokenizer:
    """
    Tokenizes text by splitting words and punctuation using ICU segmenter.
    Whitespaces will be preserved as a special token ▁ for lossless detokenization.
    Requires installation with the steps specified in https://pypi.org/project/PyICU/
    """

    def __init__(self, lang: str):
        self.lang = lang

    def tokenize(self, text:str) -> Tuple[TokenList, TokenSpanList]:
        from icu import BreakIterator, Locale

        bi = BreakIterator.createWordInstance(Locale(self.lang))
        bi.setText(text)

        tokens = []
        start = bi.first()
        for end in bi:
            token = text[start:end]
            if (
                token and token != "\n"
            ):  # exclude empty tokens, but leave whitespaces and replace them with a special token
                tokens.append(token)
            start = end

        spans: TokenSpanList = []
        offset = 0
        for token in tokens:
            offset = text.find(token, offset)
            if offset == -1:
                raise RuntimeError(f"Could not find token '{token}' in original text")
            spans.append(slice(offset, offset + len(token)))
            offset += len(token)

        tokens = [token.replace(" ", ICU_WHITESPACE_TOKEN) for token in tokens]
        return tokens, spans

class IcuDetokenizer:
    """
    Detokenizes tokens back into the original text preserving whitespaces as well.
    Spans for whitespaces will be None.
    """

    # For compatibility with MosesDetokenizer interface
    def __init__(self, lang):
        self.lang = lang

    def detokenize(self, tokens:TokenList) -> Tuple[str,TokenSpanList]:
        text = "".join(tokens).replace(ICU_WHITESPACE_TOKEN, " ")

        spans = []
        offset = 0

        for token in tokens:
            if token == ICU_WHITESPACE_TOKEN:
                spans.append(None)
                continue
            # there are some edge cases where a whitespace can appear inside a token
            token = token.replace(ICU_WHITESPACE_TOKEN, " ")
            offset = text.find(token, offset)
            if offset == -1:
                raise RuntimeError(f"Could not find token '{token}' in detokenized text")
            spans.append(slice(offset, offset + len(token)))
            offset += len(token)

        return text, spans
