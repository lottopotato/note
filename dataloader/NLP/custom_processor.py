import csv
import numpy as np
import os
import sys
import pandas as pd

#from tqdm.notebook import tqdm
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id


class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r", encoding="utf-8-sig") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				if sys.version_info[0] == 2:
					line = list(unicode(cell, 'utf-8') for cell in line)
				lines.append(line)
			return lines

## Custom processor 
class CustomProcessor(DataProcessor):
	''' Processor for various purposes:
			1. classification category (Multi label classifier)
			2. classification sentiment (binary label classifier)
	'''
	def __init__(self, text_idx, label_idx, eliminate_header = True):
		# Indicated text index, label index
		self.text_idx  		  = text_idx
		self.label_idx 		  = label_idx
		self.eliminate_header = eliminate_header
		super(CustomProcessor).__init__()

	def get_examples(self, data, quotechar):
		# file_type = 'tsv' or 'dataframe'
		if isinstance(data, str):
			data_name = data
			file_type = data_name.split('.')[-1]
		else:
			data_name = None
			file_type = 'dataframe'
		
		if file_type == 'tsv':
			data = self._read_tsv(data_name)
		elif file_type == 'json':
			data = pd.read_json(data_name)
		elif file_type == 'txt':
			try:
				data = pd.read_csv(data_name)
			except:
				data = pd.read_csv(data_name, sep = '\t')
		else:
			try:
				examples = self._create_exmples(data, quotechar, file_type)
			except:
				raise ValueError('not available data type')
		
		examples = self._create_exmples(data, quotechar, file_type)
		return examples

	def get_train_examples(self, data_name, quotechar = 'train'):
		return self.get_examples(data_name, quotechar)

	def get_valid_examples(self, data_name, quotechar = 'valid'):
		return self.get_examples(data_name, quotechar)
	def get_dev_examples(self, data_name, quotechar = 'valid'):
		return self.get_examples(data_name, quotechar)

	def get_test_examples(self, data_name, quotechar = 'test'):
		return self.get_examples(data_name, quotechar)

	def get_labels(self, label_number):
		return np.arange(label_number)


	def _create_exmples(self, lines, set_type, file_type):
		examples = []
		if file_type == 'tsv':
			iteration = enumerate(lines)
		elif file_type == 'dataframe':
			iteration = lines.iterrows()
		else:
			try:
				iteration = lines.iterrows()
			except:
				raise TypeError('Not available file type:', file_type)


		for (i, line) in tqdm(iteration, 
					desc = 'create examples - ',
					total = len(lines)):
			guid = '%s-%s' % (set_type, i)
			text_a = line[self.text_idx]
			label = line[self.label_idx]

			if isinstance(text_a, str):
				examples.append(
					InputExample(guid = guid, text_a = text_a, 
								 text_b = None, label = label))
			#print('examples -', str(i), end = '\r')
			if i<3:
				print(text_a)
			
		return examples

def convert_example_to_feature(example_row, pad_token=0, onehot_label = False,
							   sequence_a_segment_id=0, sequence_b_segment_id=1,
							   cls_token_segment_id=1, pad_token_segment_id=0,
							   mask_padding_with_zero=True, sep_token_extra=False):
	example, label_map, max_seq_length, tokenizer, onehot_label  = example_row[:5]
	cls_token_at_end, cls_token, sep_token, cls_token_segment_id = example_row[5:9]
	pad_on_left, pad_token_segment_id, sep_token_extra 			 = example_row[9:]

	tokens_a = tokenizer.tokenize(example.text_a)

	tokens_b = None
	if example.text_b:
		tokens_b = tokenizer.tokenize(example.text_b)
		# Modifies `tokens_a` and `tokens_b` in place so that the total
		# length is less than the specified length.
		# Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
		special_tokens_count = 4 if sep_token_extra else 3
		_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
	else:
		# Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
		special_tokens_count = 3 if sep_token_extra else 2
		if len(tokens_a) > max_seq_length - special_tokens_count:
			tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

	tokens = tokens_a + [sep_token]
	segment_ids = [sequence_a_segment_id] * len(tokens)

	if tokens_b:
		tokens += tokens_b + [sep_token]
		segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

	if cls_token_at_end:
		tokens = tokens + [cls_token]
		segment_ids = segment_ids + [cls_token_segment_id]
	else:
		tokens = [cls_token] + tokens
		segment_ids = [cls_token_segment_id] + segment_ids

	input_ids = tokenizer.convert_tokens_to_ids(tokens)

	# The mask has 1 for real tokens and 0 for padding tokens. Only real
	# tokens are attended to.
	input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


	# Zero-pad up to the sequence length.
	padding_length = max_seq_length - len(input_ids)
	if pad_on_left:
		input_ids = ([pad_token] * padding_length) + input_ids
		input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
		segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
	else:
		input_ids = input_ids + ([pad_token] * padding_length)
		input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
		segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

	assert len(input_ids) == max_seq_length
	assert len(input_mask) == max_seq_length
	assert len(segment_ids) == max_seq_length
	
	if onehot_label:
		label_id = example.label
	else:
		label_id = label_map[example.label]
   
	return InputFeatures(input_ids=input_ids,
						input_mask=input_mask,
						segment_ids=segment_ids,
						label_id=label_id)
	

def convert_examples_to_features(examples, label_list, max_seq_length,
								 tokenizer, onehot_label = False,
								 cls_token_at_end=False, sep_token_extra=False, pad_on_left=False,
								 sequence_a_segment_id=0, sequence_b_segment_id=1,
								 cls_token_segment_id=0, pad_token_segment_id=0,
								 mask_padding_with_zero=True,
								 process_count=cpu_count()):
	""" Loads a data file into a list of `InputBatch`s
		`cls_token_at_end` define the location of the CLS token:
			- False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
			- True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
		`cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
	"""
	cls_token = tokenizer.cls_token
	sep_token = tokenizer.sep_token
	pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
		
	if onehot_label:
		label_map = label_list
	else:
		label_map = {label : i for i, label in enumerate(label_list)}

	examples = [(example, label_map, max_seq_length, tokenizer, onehot_label, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id, sep_token_extra) for example in examples]

	with Pool(process_count) as p:
		features = list(tqdm(p.imap(convert_example_to_feature, examples, chunksize=500), total=len(examples)))

	return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()