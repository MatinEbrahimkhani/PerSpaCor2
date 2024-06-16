from enum import Enum


class Type(Enum):
    """
    An enumeration of corpus types.

    Attributes:
    sents_raw (int): A corpus of raw sentences.
    sents_tok (int): A corpus of tokenized sentences.
    whole_raw (int): A corpus of raw text.
    whole_tok (int): A corpus of tokenized text.
    """
    sents_raw = 1
    sents_tok = 2
    whole_raw = 3
    whole_tok = 4
