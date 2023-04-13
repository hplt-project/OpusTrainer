import random
from typing import List
from opustrainer.modifiers import Modifier


class PrefixModifier(Modifier):
    """Prefixes the source sentence with a phrase from the target sentence. 
    Note that if the target is ZH, JA or basically any other language that is not space segmented,
    this wouldn't work as we would be prefixing the whole sentence (or very large part of it).
    This is not supported for now. A proper fix would be to use custom detokenizer like in the placeholders.

       Usage:
       ```yaml
       modifiers:
       - Prefix: 0.02
         min_words: 2
         max_words: 5
         template: "__start__ {trg} __end__ "
        ```
    """
    min_words: int
    max_words: int
    template: str

    def __init__(self, probability: float=0.0, min_words: int=2,
        max_words: int=5, template: str="__start__ {trg} __end__ "):
        super().__init__(probability)
        self.min_words = min_words
        self.max_words = max_words
        self.template = template

    def __call__(self, line:str) -> str:
        """Takes a line in the form of "I like pie. Me gustan los pasteles."
           and turns it into: "__start__ los pasteles __end__ I like pie. Me
           gustan los pasteles."
        """
        if self.probability < random.random():
            return line
        
        # Take care of cases where we could have the alignment info at the end.
        fields: List[str] = line.split('\t')
        target_tok: List[str] = fields[1].split()

        # determine the length of the sample
        num_tokens = random.randint(self.min_words, self.max_words)

        # Select start token, accounting for the fact that we need at least num_tokens in the sentence
        max_start_token = len(target_tok) - num_tokens

        # If we have a negative max_start_token it means that the sentence
        # is too short for this augmentation and we should just skip it.
        if max_start_token < 0:
            return line
  
        start_token = random.randrange(max_start_token)
        
        augment_substring:str  = " ".join(target_tok[start_token:start_token + num_tokens])

        # Return the modified string
        return self.template.format(trg=augment_substring) + line
