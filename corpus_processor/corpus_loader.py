import os
from enum import Enum
import random

from .corpus_type import Type
from .corpus_handler import Handler
from .corpus_builder import Builder


class Loader:
    """
     A class to load corpora and build them if needed.

     Attributes
     ----------
     _filehandler : Handler
         A file handler object to read the corpus files.
     _corpus_path : dict
         A dictionary containing the corpus files.
      _tok_delim : str
         A string representing the token delimiter.
     _sent_delim : str
         A string representing the sentence delimiter.

     Methods
     -------
     load_corpus(corpus_name, corpus_type: Enum, shuffle_sentences=False, shuffle_tokens=False) -> str or list:
         Loads the corpus and shuffles it if needed.
     _load_corpus(corpus_name, corpus_type: Enum) -> str or list:
         Loads the corpus and builds it if needed.
     """

    def __init__(self, tok_delim="\b", sent_delim="\n"):
        """
        Parameters
        ----------
        tok_delim : str, optional
            A string representing the token delimiter (default is "\b").
        sent_delim : str, optional
            A string representing the sentence delimiter (default is "\n").
        """
        self._filehandler = Handler()
        self._corpus_path = {'bijankhan': self._filehandler.get_file("bijankhan_unprocessed"),
                             'peykareh': self._filehandler.get_file("peykareh_unprocessed")}

        self._tok_delim = tok_delim
        self._sent_delim = sent_delim

    def _load_corpus(self, corpus_name, corpus_type: Enum):
        """
        Loads the corpus and builds it if needed.

        Parameters
        ----------
        corpus_name : str
            The name of the corpus to be loaded.
        corpus_type : Enum
            The type of corpus to be loaded.

        Returns
        -------
        whole : str or list
            The contents of the corpus as a string or a list of strings.
        """

        if corpus_type not in list(Type):
            raise Exception("invalid corpus type requested")
        file_key = Handler.get_file_key(corpus_name, corpus_type)

        if not os.path.isfile(self._filehandler.get_file(file_key)):
            print(f"Requested version not found corpus_name: {corpus_name}, corpus_type: {corpus_type}")
            print("building the corpus")
            return Builder().build_corpus(corpus_name, corpus_type)

        file_path = self._filehandler.get_file(file_key)
        if corpus_type.value == Type.whole_raw.value:
            with open(file_path, "r") as f:
                # Reading the lines from the file
                whole = f.read()
            return whole
        elif corpus_type.value == Type.whole_tok.value:
            with open(file_path, "r") as f:
                # Reading the lines from the file
                whole = f.read()
                whole = whole.split(self._tok_delim)
            return whole
        elif corpus_type.value == Type.sents_raw.value:
            with open(file_path, "r") as f:
                # Reading the lines from the file
                whole = f.read()
                whole = whole.split(self._sent_delim)
            return whole
        elif corpus_type.value == Type.sents_tok.value:
            with open(file_path, "r") as f:
                # Reading the lines from the file
                whole = f.read()
                whole = whole.split(self._sent_delim)
                whole = [sentence.split(self._tok_delim) for sentence in whole]
            return whole

    def load_corpus(self, corpus_name, corpus_type: Enum, shuffle_sentences=False):
        """
        Loads the corpus and shuffles it if needed.

        Parameters
        ----------
        corpus_name : str
            The name of the corpus to be loaded.
        corpus_type : Enum
            The type of corpus to be loaded.
        shuffle_sentences : bool, optional
            A boolean indicating whether to shuffle the sentences (default is False).

        Returns
        -------
        loaded : str or list
            The contents of the corpus as a string or a list of strings.
        """
        if corpus_name in self._filehandler.corpus_names():
            loaded = self._load_corpus(corpus_name, corpus_type)
        elif corpus_name == 'all':
            if corpus_type.value == Type.whole_raw.value:
                loaded = ""
            else:
                loaded = []
            if shuffle_sentences:
                loaded = []
                shuffle_sentences = False
                for name in self._filehandler.corpus_names():

                    loaded += self._load_corpus(name, Type.sents_raw)
                random.shuffle(loaded)

            else:
                for name in self._filehandler.corpus_names():
                    loaded += self._load_corpus(name, corpus_type)
        else:
            raise Exception(
                f'invalid corpus name \nrequested name:\t{corpus_name} \navailable names:'
                f'\t{self._filehandler.corpus_names()}')

        if shuffle_sentences and (
                corpus_type.value == Type.sents_tok.value or corpus_type.value == Type.sents_raw.value):
            random.shuffle(loaded)

        elif shuffle_sentences:
            raise Exception("Invalid shuffling strategy")
        return loaded
