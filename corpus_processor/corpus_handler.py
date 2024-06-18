import csv

import os
from enum import Enum
from .corpus_type import Type


class Handler:
    def __init__(self, base_directory="./", filenames="_data/_filenames.csv"):
        """
        Initializes the CorpusHandler class with a directory to search and a file that contains the filenames and
        their paths.

        Parameters: directory (str): The directory to search for files. Default is "./". filenames (str): The path to
        the file that contains the filenames and their paths. Default is "./corpus/_filenames.csv".
        """
        self.directory = base_directory
        self._corpus_names = ['bijankhan', 'peykareh']
        self._corpus_types = [e for e in Type]
        self.filenames = self.directory+filenames
        self._files = {}
        self._load_filenames()

    def _load_filenames(self):
        """
        Loads the filenames and their paths from the file specified in the filenames attribute.
        """
        try:
            with open(self.filenames, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    self._files[row[0]] = self.directory+row[1]
        except Exception as e:
            print("unable to find filenames.csv", e)

    def get_file(self, file_key):
        """
        Returns the path of the file with the specified file_key.

        Parameters:
        file_key (str): The key of the file to retrieve.

        Returns:
        str: The path of the file with the specified file_key.

        Raises:
        Exception: If the specified file_key is not in the dictionary of files.
        """
        return self._files.get(file_key,
                               f"File '{file_key}' not found. Available filenames: {list(self._files.keys())}")

    @staticmethod
    def get_file_key(corpus_name, corpus_type: Enum):
        """
        Returns a string that represents the key for a corpus file.

        :param corpus_name: The name of the corpus.
        :type corpus_name: str
        :param corpus_type: The type of the corpus.
        :type corpus_type: Enum
        :return: A string that represents the key for a corpus file.
        :rtype: str
        """
        return corpus_name + "_" + str(corpus_type.name)

    def find_files(self, name):
        """
        Finds files in the directory and subdirectories that match the specified name and updates the dictionary with
        the files and their paths. Parameters: name (str): The name of the files to search for.
        """
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file == name:
                    self._files[file] = os.path.join(root, file)

    @property
    def files(self):
        """
        Returns the dictionary of files and their paths.

        Returns:
        dict: The dictionary of files and their paths.
        """
        return self._files

    def corpus_names(self):
        return self._corpus_names

    def corpus_types(self):
        return list(self._corpus_types)

    @staticmethod
    def split_file(file_path):
        """
        Splits a file into smaller chunks of 64 MB each.

        Parameters:
        file_path (str): The path to the file to be split.

        Returns:
        None
        """
        max_size = 64 * 1024 * 1024  # 100 MB
        with open(file_path, 'rb') as f:
            chunk = f.read(max_size)
            chunk_num = 1

            while chunk:
                with open(f'{os.path.splitext(file_path)[0]}{chunk_num}.txt', 'wb') as out_file:
                    out_file.write(chunk)

                chunk = f.read(max_size)
                chunk_num += 1

    @staticmethod
    def combine_files(file_path):
        """
       Combines the split files into a single file.

       Parameters:
       file_path (str): The path to the file to be combined.

       Returns:
       None
        """
        with open(file_path, 'wb') as outfile:
            chunk_num = 1

            while os.path.exists(f'{os.path.splitext(file_path)[0]}{chunk_num}.txt'):
                with open(f'{os.path.splitext(file_path)[0]}{chunk_num}.txt', 'rb') as infile:
                    outfile.write(infile.read())
                chunk_num += 1
