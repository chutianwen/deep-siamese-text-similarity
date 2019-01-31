from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
from tensorflow.contrib.learn.python import learn  # pylint: disable=g-bad-import-order

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",re.UNICODE)


def tokenizer_char(iterator):
	for value in iterator:
		yield list(value)


def tokenizer_word(iterator):
	for value in iterator:
		yield TOKENIZER_RE.findall(value)


class MyVocabularyProcessor(learn.preprocessing.VocabularyProcessor):
	def __init__(self,
	             max_document_length,
	             min_frequency=0,
	             vocabulary=None,
	             is_char_based=True):
		if is_char_based:
			tokenizer_fn = tokenizer_char
		else:
			tokenizer_fn = tokenizer_word

		super().__init__(max_document_length, min_frequency, vocabulary, tokenizer_fn)
