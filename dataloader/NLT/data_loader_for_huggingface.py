import os
import sys
import pandas as pd
import numpy as np

import torch
import torch.utils.data as torch_data
import torchtext
torchtext_version = list(map(int, torchtext.__version__.split('.')))
if torchtext_version[0] <= 0 and torchtext_version[1] < 9:
	from torchtext import data, datasets
else:
	from torchtext.legacy import data, datasets

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from ast import literal_eval
from tqdm import tqdm

class NLT_dataset(torch_data.Dataset):
	def __init__(self, data_path, src_vocab_path, tgt_vocab_path, 
				 unk_token = '<UNK>', pad_token = '<PAD>',
				 bos_token = '<BOS>', eos_token = '<EOS>',
				 special_tokens = {'normal':'<NOR>', 'law':'<LAW>', 'patent':'<PTT>'},
				 max_length = None,
				 domain_column = 'type',
				 return_tensors = 'pt',
				 verbose = None, dual_learning = False,
				 column_names = None, sampling = None):
		
		super(NLT_dataset, self).__init__()
		self.unk_token 	   	= unk_token
		self.pad_token 	   	= pad_token
		self.bos_token 	   	= bos_token
		self.eos_token 	   	= eos_token
		self.src_vocab 	   	= AutoTokenizer.from_pretrained(src_vocab_path)
		self.tgt_vocab 	   	= AutoTokenizer.from_pretrained(tgt_vocab_path)
		self.src_bos_id		= self.src_vocab.bos_token_id
		self.src_eos_id 	= self.src_vocab.eos_token_id
		self.src_pad_id 	= self.src_vocab.pad_token_id
		self.tgt_bos_id 	= self.tgt_vocab.bos_token_id
		self.tgt_eos_id 	= self.tgt_vocab.eos_token_id
		self.tgt_pad_id 	= self.tgt_vocab.pad_token_id


		self.verbose 	   	= verbose
		self.dual_learning 	= dual_learning
		self.column_names  	= column_names
		self.sampling 	   	= sampling
		self.special_tokens = special_tokens
		self.domain_column 	= domain_column
		self.return_tensors = return_tensors
		self.max_length		= max_length

		if isinstance(data_path, list) and special_tokens is not None:
			data_list = []
			for path_ in data_path:
				for domain, domain_token in self.special_tokens.items():
					if os.path.basename(path_).split('.')[1] == domain:
						data_list.append(self.processing(path_, domain))
			self.dataset_df	= pd.concat(data_list)
			self.set_domain = True
		else:
			self.dataset_df	= self.processing(data_path)
			self.set_domain = False

	def processing(self, data_path, domain = None, init_column_names = ['ko', 'en']):
		print('read dataset - ', str(data_path))
		dataset = pd.read_csv(data_path, sep = '\t', names = init_column_names)
		if domain is not None:
			dataset[self.domain_column] = domain
		if self.sampling is not None:
			if isinstance(self.sampling, int):
				dataset = dataset[:self.sampling]
			elif isinstance(self.sampling, float) and self.sampling < 1:
				dataset = dataset[:int(len(dataset) * self.sampling)]
			else:
				raise AttributeError('sampling must be an integer or a number less than or equal to 1.')

		print('#: ', len(dataset))
		return dataset
	
	def getdata(self, df, vocab, i, lang_name):
		truncation = True if self.max_length else False
		if self.set_domain:
			data = vocab(df.iloc[i][lang_name],
				return_tensors = self.return_tensors, add_special_tokens = False, 
				return_length = True, return_token_type_ids=False, 
				max_length = self.max_length, truncation = truncation,
				return_attention_mask=False), vocab.convert_tokens_to_ids(
					self.special_tokens[df.iloc[i][self.domain_column]])
		else:
			data = vocab(df.iloc[i][lang_name], 
				return_tensors = self.return_tensors, add_special_tokens = False, 
				return_length = True, return_token_type_ids=False, 
				max_length = self.max_length, truncation = truncation,
				return_attention_mask=False), None
			
		return data

	def __getitem__(self, i):
		return (self.getdata(self.dataset_df, self.src_vocab, i, self.column_names[0]),
			self.getdata(self.dataset_df, self.tgt_vocab, i, self.column_names[1]))
		
	def __len__(self):
		return len(self.dataset_df)
	
	def __iter__(self):
		return None
	
	def _get_vocab(self):
		return self.src_vocab, self.tgt_vocab

def tensor_transform(data, bos_id, eos_id, set_ebos = False, set_domain = False):
	data_tensor = data[0].input_ids[0]
	data_length = data[0].length
	data_domain = data[1]
	#print(data_tensor, torch.tensor([bos_id]))
	#print(data_tensor.shape, torch.tensor([bos_id]).shape)


	if set_ebos:
		data = torch.cat((torch.tensor([bos_id]),
			data_tensor, 
			torch.tensor([eos_id])))
		length = data_length + 2
	else:
		data   = data_tensor
		length = data_length

	if set_domain and not data_domain is None:
		data = torch.cat((torch.tensor([data_domain]),
			data))
		length += 1

	return data, length

def batch_transform(batch, src_pad_id, src_bos_id, src_eos_id,
					tgt_pad_id, tgt_bos_id, tgt_eos_id, 
					set_ebos = False, set_domain = False):
	src_batch, tgt_batch = [], []
	src_length, tgt_length = [], []
	# eos bos add
	for src_ids, tgt_ids in batch:
		src_tensor = tensor_transform(src_ids, bos_id = src_bos_id, 
			eos_id = src_eos_id, set_ebos = set_ebos, set_domain = set_domain)
		tgt_tensor = tensor_transform(tgt_ids, bos_id = tgt_bos_id,
			eos_id = tgt_eos_id, set_ebos = True, set_domain = False)
		src_batch.append(src_tensor[0])
		src_length.append(src_tensor[1])
		tgt_batch.append(tgt_tensor[0])
		tgt_length.append(tgt_tensor[1])
	src_batch = pad_sequence(src_batch, padding_value = src_pad_id, batch_first = True)
	tgt_batch = pad_sequence(tgt_batch, padding_value = tgt_pad_id, batch_first = True)
	
	src_length = torch.tensor(src_length)
	tgt_length = torch.tensor(tgt_length)	
	
	# sorting for descending
	src_length, src_indices = torch.sort(src_length, descending = True)

	return src_batch[src_indices], src_length, tgt_batch[src_indices], tgt_length[src_indices]

class Collate_fn:
	def __init__(self, dataset):
		self.src_bos_id = dataset.src_bos_id
		self.src_eos_id = dataset.src_eos_id
		self.src_pad_id = dataset.src_pad_id
		self.tgt_bos_id = dataset.tgt_bos_id
		self.tgt_eos_id = dataset.tgt_eos_id
		self.tgt_pad_id = dataset.tgt_pad_id
		#print(self.src_eos_id, self.src_bos_id, self.src_pad_id)
		#print(self.tgt_eos_id, self.tgt_bos_id, self.tgt_pad_id)
		self.set_ebos 	= dataset.dual_learning
		self.set_domain = dataset.set_domain
	
	def __call__(self, batch):
		src_batch, src_length, tgt_batch, tgt_length = batch_transform(
			batch = batch, 
			src_pad_id = self.src_pad_id,
			src_bos_id = self.src_bos_id, 
			src_eos_id = self.src_eos_id, 
			tgt_pad_id = self.tgt_pad_id,
			tgt_bos_id = self.tgt_bos_id, 
			tgt_eos_id = self.tgt_eos_id,
			set_ebos = self.set_ebos,
			set_domain = self.set_domain)
		return ((src_batch, src_length), (tgt_batch, tgt_length))
		


class NLT_DataLoader(DataLoader):
	def __init__(self, data_path, 
				 src_vocab_path, tgt_vocab_path,
				 batch_size,
				 shuffle = False, max_length = None,
				 unk_token = '<UNK>', pad_token = '<PAD>',
				 bos_token = '<BOS>', eos_token = '<EOS>',
				 special_tokens = {'normal':'<NOR>', 'law':'<LAW>', 'patent':'<PTT>'},
				 domain_column = 'type',
				 return_tensors = 'pt',
				 verbose = None, dual_learning = False,
				 column_names = None, sampling = None,
				 num_workers = 20
				 ):

		args = (data_path, src_vocab_path, tgt_vocab_path,
				unk_token, pad_token, bos_token, eos_token, 
				special_tokens, max_length, domain_column, return_tensors, 
				verbose, dual_learning, column_names, sampling)
		
		self.dataset = NLT_dataset(*args)
		collate_fn   = Collate_fn(self.dataset)
		super().__init__(self.dataset, 
						 shuffle = shuffle, 
						 batch_size = batch_size, 
						 collate_fn = collate_fn,
						 num_workers = num_workers)



