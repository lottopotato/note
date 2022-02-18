# python3.8
"""
	torch(huggingFace) model training wrapper
"""

import os
import numpy as np
import math
import sys

## torch >=1.9.0
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


class Trainer:
	def __init__(self, num_labels, task, model, tokenizer, 
		pretrained_model, environment = 'shell'):
		self.task_list = ['binary-class', 'multi-class', 'multi-label', 'binary']
		"""
			3 major types of deeplearning classes:
			 1. binary-class: label is one of two labels.
			 2. multi-class: label is one of various labels.
			 3. multi-label: labels are one or more of various labels.
		"""
		if not task in self.task_list:
			raise RuntimeError('task should be either one of ', str(self.task_list))
		if not environment in ['shell', 'jupyter']:
			raise RuntimeError('environment should be either one of \'jupyter\' or \'shell\'')
		if environment == 'shell':
			from tqdm import tqdm
		else:
			from tqdm.notebook import tqdm
		self.task = task
		self.tqdm = tqdm

		self.device	= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model.to(self.device)

	def save_model(self, save_name, key = None, model_dict = None):
		if key:
			if model_dict:
				model_state_dict = model_dict.update({key: self.model.state_dict()})
			else:
				model_state_dict = {key: self.model.state_dict()}
		else:
			model_state_dict = self.model.state_dict()
		torch.save(model_state_dict, save_name)

	def load_model(self, save_name, key = None):
		if not os.path.exists(save_name):
			raise FileNotFoundError(f'not found file: {save_name}')
		if key:
			self.model.load_state_dict(torch.load(save_name)[key])
		else:
			self.model.load_state_dict(torch.load(save_name))

	def get_dataset(self, dataset, onhot_label, get_func):
		features = get_func(dataset)

		all_input_ids         = torch.tensor([f.input_ids for f in features], dtype = torch.long)
		all_input_mask        = torch.tensor([f.input_mask for f in features], dtype = torch.long)
		all_input_segment_ids = torch.tensor([f.segment_ids for f in features], dtype = torch.long)
		
		if self.task == 'multi-class':
			label_dtype = torch.float
		elif self.task == 'binary':
			label_dtype = torch.long
		else:
			raise KeyError('task should be either one of ', str(self.task_list))
		all_label_ids = torch.tensor([f.label_id for f in features], dtype = label_dtype)    

		dataset = TensorDataset(all_input_ids, all_input_mask, all_input_segment_ids, all_label_ids)
		return dataset

	def prepareData(self, get_train_processor, get_valid_processor = None, get_test_processor = None,
		label_list, max_seq_length, dataset_path,  
		batch_size, data_dir = '', training = True, onehot_label = False):
		self.max_seq_length = max_seq_length

		train_set, valid_set, test_set = None, None, None

		if isinstance(dataset_path, str):
			print(f'{dataset_path} is alone, it will use for test.')
			training = False
			test_set = dataset_path
		else:
			if len(dataset_path) > 3:
				raise ValueError('can not recognize dataset path.')
			elif len(dataset_path) == 3:
				print(f'if dataset path are 3, must be ordered statically to [train set, valid set, test set]')
				train_set, valid_set, test_set = dataset_path
			elif len(dataset_path) == 2:
				print(f'if dataset path are 2, must be ordered statically to [train set, valid set]')
				train_set, valid_set = dataset_path
			elif len(dataset_path) == 1:
				print(f'{dataset_path} is alone, it will use for test.')
				test_set = dataset_path
				training = False
		
		self.label_list = label_list

		self.train_dataloader = None
		self.valid_dataloader = None

		if training and train_set:
			train_examples = get_train_processor(train_set)
			train_dataset  = self.get_dataset(train_examples, onehot_label)
			
			self.train_dataloader = DataLoader(
				train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)

		if valid_set and get_valid_processor:
			valid_examples = get_valid_processor(valid_set)
			valid_dataset  = self.get_dataset(valid_examples, onehot_label)

			self.valid_dataloader = DataLoader(
				valid_dataset, sampler = SequentialSampler(valid_dataset), batch_size = batch_size)

		if test_set and get_test_processor:
			test_examples = get_test_processor(test_set)
			test_dataset  = self.get_dataset(test_examples, onehot_label)

			self.test_dataloader = DataLoader(
				test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size)

	def get_accuracy(self, y_pred, y_true, metric_name = None):
		def binary_accuracy(y_pred, y_true):
			if not len(y_true.shape) == 1:
				top_prediction = np.where([y_pred[i] == y_pred[
					i, np.argmax(y_pred, axis = 1)] for i in range(y_pred.shape[0])], 1, 0)
				y_pred = top_prediction.flatten()
				f1_avg = 'micro'
			else:
				y_pred = y_pred.argmax(axis = 1)
				f1_avg = 'binary'
			y_true = y_true.flatten()
			return f1_score(y_true, y_pred, average = f1_avg)
		def hit1_accuracy(y_pred, y_true):
			if len(y_true.shape) == 1:
				return binary_accuracy(y_pred, y_true)
			else:
				pred = np.argmax(y_pred, 1)
				hit1 = y_true[np.arange(y_true.shape[0]), pred]
				return np.average(hit1)
		def f1_round_accracy(y_pred, y_true):
			if not len(y_true.shape) == 1:
				y_pred = 1/(1 + np.exp(-y_pred))
				y_pred = y_pred.round().flatten()
				f1_avg = 'micro'
			else:
				y_pred = y_pred.argmax(axis = 1)
				f1_avg = 'binary'
			y_true = y_true.flatten()
			return f1_score(y_true, y_pred, average = f1_avg)
		def flat_accuracy(y_pred, y_true):
			y_pred = np.argmax(y_pred, axis = 1).flatten()
			y_true = y_true.flatten()
			return np.sum(y_pred == y_true)/len(y_true)

		if metric_name == 'binary':
			metric = binary_accuracy
		elif metric_name == 'hit1':
			metric = hit1_accuracy
		elif metric_name == 'multi' or metric_name == 'multi-classes':
			metric = f1_round_accracy
		elif metric_name == 'flat_accuracy' or metric_name == 'single-label':
			metric = flat_accuracy
		else:
			metric = torch.metrics.Accuracy()

		y_pred = y_pred.cpu().numpy()
		y_true = y_true.cpu().numpy()
		return metric(y_pred, y_true)

	@staticmethod
	def get_batch(batch, device, x_indices, y_index = None):
		batch   = [t.to(device) for t in batch]
		x_batch = [batch[i] for i in x_indices]
		if y_index:
			return x_batch, batch[y_index]
		else:
			return x_batch

	def prediction(self, get_test_processor, dataset, label_list, batch_size, max_seq_length,
		x_indices,
		file_type = 'eager', as_array = True, actiavtion = None):
		examples   = get_test_processor(file)
		dataset    = self.get_dataset(examples)
		dataloader = DataLoader(
			dataset, SequentialSampler(dataset), batch_size)

		if activation == 'sigmoid':
			activation = torch.nn.functional.sigmoid
		elif activation == 'softmax':
			activation = torch.nn.functional.softmax
		else:
			activation = activation

		self.model.eval()
		prediction_list = []
		print('\n predction..')
		for step, batch in self.tqdm(enumerate(dataloader), desc = 'steps', total = len(dataloader)):
			with torch.no_grad():
				batch = self.get_batch(batch, self.device, x_indices)
				y_pred = self.model(*batch)
				if activation:
					y_pred = activation(y_pred)
				predction_list.extend(y_pred.cpu().numpy())
		return predction_list if not as_array else np.as_array(prediction_list)

	def evaluate(self, dataloader, x_indices, y_index, loss_fn, metric, data_name = 'validation'):
		loss, acc = 0, 0
		self.model.eval()
		for step, batch in self.tqdm(enumerate(dataloader), desc = 'steps', total = len(dataloader)):
			with torch.no_grad():
				X, y = self.get_batch(batch, self.device, x_indices, y_index)

				y_pred = self.model(*X).logits

				batch_loss = loss_fn(y_pred, y)
				batch_acc  = self.get_accuracy(y_pred, y, metric)

				loss += batch_loss.item()
				acc  += batch_acc.item()
		print('{0} loss: {1:.8f} {0} acc {2:.8f}'.format(
			data_name, loss/(step+1), acc/(step+1)))
		return loss/(step+1), acc/(step+1)

	def training(self, epochs, optimizer, metric, loss_fn,
		scheduler = None, eval_valid = True):
		if eval_valid and self.valid_dataloader is None:
			raise RuntimeError('eval_valid can not be True without valid dataset.')

		train_loss_list, train_acc_list = [], []
		if eval_valid:
			valid_loss_list, valid_acc_list = [], []
		for epochs in self.tqdm(range(epochs), desc = 'epochs', total = epcohs):
			train_loss, train_acc = 0, 0
			self.model.train()
			for step, batch in self.tqdm(enumerate(self.train_dataloader),
				desc = 'steps', total = len(self.train_dataloader)):
				optimizer.zero_grad()

				X, y = self,get_batch(batch, self.device, x_indices, y_index)

				logits = self.model(*X).logits

				train_batch_loss = loss_fn(logits, y)
				train_batch_loss.backward()

				optimizer.step()
				if scheduler:
					scheduler.step()

				with torch.no_grad():
					train_batch_acc = self.get_accuracy(logits, y, metric)
				train_loss += train_batch_loss.item()
				train_acc  += train_batch_acc.item()
				print('E-{0} train loss: {1:.8f} train acc: {2:.8f}'.format(
						epoch, train_loss/(step+1), train_acc/(step+1)), end = '\r')
			train_loss /= (step+1)
			train_acc  /= (step+1)

			train_loss_list.append(train_loss)
			train_acc_list.append(train_acc)

			if eval_valid:
				valid_loss, valid_acc = self.evaluate(self.valid_dataloader)
				valid_loss_list.append(valid_loss)
				valid_acc_list.append(valid_acc)

		results = {'train_loss': train_loss_list, 'train_acc': train_acc_list}
		if eval_valid:
			results.update({'valid_loss': valid_loss_list, 'valid_acc': valid_acc_list})

		return results	








