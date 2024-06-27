import re
import numpy as np
import sys

import warnings

from corpus_processor import Loader
from corpus_processor import Type as Ctype

# Set the print option for numpy arrays to display the whole array without truncation
np.set_printoptions(threshold=sys.maxsize)


class Labeler:
    """A class that labels text based on regular expressions.

    Attributes:
        _tags (tuple): A tuple of tags for each class, such as ('1', '2').
        _regexes (tuple): A tuple of regular expressions for each class, such as (r'[^\S\r\n\v\f]', r'\u200c').
        _class_chars (tuple): A tuple of characters to insert after each class, such as (' ', '').
        class_count (int): The number of classes to label.
        data (str or list): The input text to be labeled, either a single string or a list of sentences.

        labels (list): A list of lists of labels for each character.
        corpus_type (Enum): An enum value indicating the type of the input text,either CorpusType.whole_raw or CorpusType.sents_raw.

    Methods:
        set_text(textinput, corpus_type): A class method that sets the data attribute of the class based on the
        input type. _sent_labeler(sent): A private method that labels a single sentence and returns a list of characters
    and a list of labels.
        _text_labeler(): A private method that labels the whole text and returns a list of
    characters and a list of labels.
        labeler(): A public method that calls either _sent_labeler or _text_labeler
    depending on the corpus_type and returns labels attributes. _text_gen(chars, labels): A private
    method that generates a labeled text from a list of characters and a list of labels by inserting the class
    characters.
        text_generator(corpus_type, chars, labels): A public method that calls _text_gen for each sentence or
    the whole text depending on the corpus_type and returns a labeled text or a list of labeled sentences.
    """

    def __init__(self, tags=(1, 2),
                 regexes=(r'[^\S\r\n\v\f]', r'\u200c'),
                 chars=(" ", "‌"),
                 class_count=2,
                 ):
        """Initialize a Labeler object with the given parameters.

        Parameters:
        tags (tuple): A tuple of tags for each class, such as (1, 2).
        regexes (tuple): A tuple of regular expressions for each class, such as (r'[^\S\r\n\v\f]', r'\u200c').
        chars (tuple): A tuple of characters to insert after each class, such as (' ', '').
        class_count (int): The number of classes to label.

        """
        self._tags = tags
        self._regexes = regexes
        self._class_chars = chars
        self.class_count = class_count

        self.data = None
        self.labels = []
        self.corpus_type = None

    # Define a class method that sets the data attribute of the class based on the input type

    def _sent_labeler(self, sent: str):
        # Initialize an empty list to store the labels
        labels = []
        # Convert the input sentence into a list of characters for the output
        characters = list(sent)
        # Initialize an empty list to store the indices of characters to be deleted
        deletable = []
        # Assign a label of "0" to each character in the sentence
        labels += [0] * len(sent)
        # Loop through the classes
        for i in range(self.class_count):
            # Find all the matches of the regular expression for the current class in the sentence
            for match in re.finditer(self._regexes[i], sent):
                # Get the index of the match. ps. it's in character level, so we don't need the end
                idx = match.start()
                # Assign the corresponding tag to the label of the character before the match
                labels[idx - 1] = self._tags[i]
                # Add the index of the match to the list of deletable
                deletable.append(idx)
        # Sort the deletable in descending order. not to mess with the indices
        deletable = sorted(deletable, reverse=True)

        # Loop through the deletable and delete the corresponding
        for deletable in deletable:
            characters.pop(deletable)
            labels.pop(deletable)

        return characters, labels

    def _text_labeler(self):

        # Initialize an empty list to store the labels
        labels = []
        # Convert the input sentence into a list of characters for the output
        # Convert the list to a dict to speed up the process
        characters = dict(enumerate(list(self.data)))
        # Initialize an empty list to store the indices of characters to be deleted
        deletable = []
        # Assign a label of "0" to each character in the sentence
        labels = [0] * len(self.data)
        # Convert the list to a dict to speed up the process
        labels = dict(enumerate(labels))
        # Loop through the classes
        for i in range(self.class_count):
            # Find all the matches of the regular expression for the current class in the sentence
            for match in re.finditer(self._regexes[i], self.data):
                # Get the index of the match. ps. it's in character level, so we don't need the end
                idx = match.start()
                # Assign the corresponding tag to the label of the character before the match
                labels[idx - 1] = self._tags[i]
                # Add the index of the match to the list of deletable
                deletable.append(idx)

        # Loop through the deletable and delete the corresponding
        for d in deletable:
            del characters[d]
            del labels[d]

        # Changing back data structure to list
        return list(characters.values()), list(labels.values())

    def _labeler(self) -> (list, list):

        # Initialize empty lists to store the characters and labels
        result_chars = []
        result_labels = []
        # Check if the data attribute is a list of sentences
        if self.corpus_type.value == Ctype.sents_raw.value:
            for sent in self.data:
                # Call the _sent_labeler method to get the characters and labels for each sentence
                characters, labels = self._sent_labeler(sent)
                # Append the characters and labels to the result lists
                result_chars.append(characters)
                result_labels.append(labels)

        # Check if the data attribute is a single string
        elif self.corpus_type.value == Ctype.whole_raw.value:
            # Call the _text_labeler method to get the characters and labels for the whole string
            result_chars, result_labels = self._text_labeler()

        return result_chars, result_labels

    def label_text(self, textinput, corpus_type: Ctype):

        if corpus_type.value == Ctype.whole_raw.value:
            type(textinput)
            if type(textinput) == str:
                # Set the data attribute to the input string
                self.data = textinput

                self.corpus_type = corpus_type
                print("input data is in str format")

        elif corpus_type.value == Ctype.sents_raw.value and type(textinput) == list:
            self.data = textinput
            self.corpus_type = corpus_type
            print("input data is in list format, processing it as in a list of sentences")

        else:
            raise Exception("invalid input for the labeler check the code")
        return self._labeler()

    def _text_generator(self, chars, labels):
        """

        :param chars: a list of characters that are together.
        :param labels: a list of labels that are in sync with cars.
        :return: returns the joined string generated from the chars and labels.
        """
        result = []
        for char, label in zip(chars, labels):
            if label == 0:
                result.append(char)
                continue

            for i in range(self.class_count):
                if label == self._tags[i]:
                    result.append(char)
                    result.append(self._class_chars[i])

        return ''.join(result)

    def text_generator(self, chars, labels, corpus_type: Ctype):
        if corpus_type.value == Ctype.sents_raw.value:
            result = []
            for sent_chars, sent_labels in zip(chars, labels):
                result.append(self._text_generator(sent_chars, sent_labels))
            return result
        elif corpus_type.value == Ctype.whole_raw.value:
            return self._text_generator(chars, labels[0])


