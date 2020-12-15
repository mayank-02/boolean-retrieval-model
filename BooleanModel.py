import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from query import (
    infix_to_postfix,
    is_left_bracket,
    is_operator,
    is_right_bracket,
)
from Stack import Stack


class BooleanModel(object):
    """ Boolean model for unranked retrieval of information """

    def __init__(self, path):

        # Path to corpus of documents
        self.CORPUS = path

        # Set of english stopwords to exclude while preprocessing
        self.STOPWORDS = set(stopwords.words("english"))

        # For stemming purposes
        self.STEMMER = PorterStemmer()

        # Postings is a dict from term to list of doc_ids it appears in
        # Ex. postings["pikachu"] = [1, 5]
        self.postings = defaultdict(list)

        # Documents is a dictionary from doc_id to document name
        self.documents = dict()

        # Set of all unique terms in the corpus
        self.vocabulary = set()

        # Preprocess the corpus and keep it ready for query
        self.__preprocessing()

    def __preprocessing(self):
        """Preprocess the corpus"""

        idx = 1

        # For every document in the corpus
        for filename in glob.glob(self.CORPUS):
            # print(filename)

            # Read the document
            with open(filename, "r") as file:
                text = file.read()

            # Remove all special characters from the text
            text = self.remove_special_characters(text)

            # Remove digits from the text
            text = self.remove_digits(text)

            # Tokenize text into words
            words = word_tokenize(text)

            # Remove stopwords and convert remaining words to lowercase
            words = [word.lower() for word in words if word not in self.STOPWORDS]

            # Stemming
            words = [self.STEMMER.stem(word) for word in words]

            # Retain only unique words
            terms = self.unique(words)

            # Add posting to postings of the term
            for term in terms:
                self.postings[term].append(idx)

            # Make a list of indexed documents
            self.documents[idx] = os.path.basename(filename)

            # Increment count of indexed documents
            idx = idx + 1

        # Make vocabulary out of final postings
        self.vocabulary = self.postings.keys()

        return

    def query(self, query):
        """Query the indexed documents using a boolean model

        :param query: valid boolean expression to search for
        :returns: list of matching document names
        """
        # Tokenize query
        query_tokens = word_tokenize(query)

        # Convert infix query to postfix
        query_tokens = infix_to_postfix(query_tokens)

        # Evaluate query against already processed documents
        matching_docs = self.__evaluate_query(query_tokens)

        return matching_docs

    def __evaluate_query(self, query_tokens):
        """Evaluates the query against the corpus

        :param query_tokens: list of query tokens in postfix form
        :returns: list of matching document names
        """

        operands = Stack()

        for token in query_tokens:

            # Token is an operator,
            # Pop two elements from stack and apply it.
            if is_operator(token):
                # Pop right operand
                right_operand = operands.pop()

                # Pop left operand
                left_operand = operands.pop()

                # Perform operation
                result = self.__perform_operation(left_operand, right_operand, token)

                # Push result back into the stack
                operands.push(result)

            # Token is an operand, push it to the stack
            else:
                # Lowercasing and stemming query term
                token = self.STEMMER.stem(token.lower())

                # Push it's bit vector into operand stack
                operands.push(self.__bitvector(token))

        if len(operands) != 1:
            print("Malformed query or postfix expression")
            return list()

        # Find out documents corresponding to set bits in the vector
        matching_docs = [self.documents[i + 1] for i in np.where(operands.peek())[0]]

        return matching_docs

    def __bitvector(self, word):
        """Make bitvector out of a word

        :param word: word
        :returns: bit vector of word with bits set when it appears in the particular documents
        """
        # Size of bit vector
        doc_count = len(self.documents)

        negate = False

        # If word is "~good"
        if word[0] == "~":
            negate = True
            word = word[1:]

        if word in self.vocabulary:
            # Word is in corpus, so make a bit vector for it
            bit_vector = np.zeros(doc_count, dtype=bool)

            # Locate query token in the dictionary and retrieve its postings
            posting = self.postings[word]

            # Set bit for doc_id in which query token is present
            for doc_id in posting:
                bit_vector[doc_id - 1] = True

            if negate:
                # Instance of NOT token,
                # bit vector is supposed to be inverted
                bit_vector = np.invert(bit_vector)

            # Return bit vector of the word
            return bit_vector

        else:
            # Word is not in corpus
            print(
                "Warning:",
                word,
                "was not found in the corpus!",
            )
            return np.zeros(doc_count, dtype=bool)

    def __perform_operation(self, left, right, op):
        """Performs specified operation on the vectors

        :param left: left operand
        :param right: right operand
        :param op: operation to perform
        :returns: result of the operation
        """

        if op == "&":
            return left & right

        elif op == "|":
            return left | right

        else:
            return 0

    def unique(self, words):
        """Removes duplicates from a list of words"""

        # Return a list of unique words
        return list(set(words))

    def remove_special_characters(self, text):
        """Removes special characters from a blob of text"""

        # Regex pattern for a word
        regex = re.compile(r"[^a-zA-Z0-9\s]")

        # Replace and return
        return re.sub(regex, "", text)

    def remove_digits(self, text):
        """Removes digits from a blob of text"""

        # Regex pattern for a word
        regex = re.compile(r"\d")

        # Replace and return
        return re.sub(regex, "", text)
