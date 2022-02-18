# python3.8
"""
	torch(huggingFace) model training by ignite
	ref: https://github.com/kh-kim/subword-nmt

"""
import numpy as np
import datetime
import os

import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils.utils import get_grad_norm, get_parameter_norm


VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

DEBUG = False
## nan, inf debug
torch.autograd.set_detect_anomaly(DEBUG)

class NLT_Engine(Engine):
	def __init__(self, func, model, loss_fn, optimizer, lr_scheduler, config,
		huggingfaceUse = False):
		self.model 		  = model
		self.loss_fn 	  = loss_fn
		self.optimizer 	  = optimizer
		self.lr_scheduler = lr_scheduler
		self.loss_div	  = config.loss_div
		self.config 	  = config
		self.lr  		  = 0
		self.record_lr 	  = True

		super().__init__(func)

		self.best_loss = 1e+3
		self.scaler	= GradScaler()

		self.scores_list = [[], [], [], []]

		self.huggingfaceUse = huggingfaceUse
		
	@staticmethod
	def train(engine, mini_batch):
		engine.model.train()
		if engine.state.iteration == 1:
			engine.optimizer.zero_grad()
		if engine.state.iteration % engine.config.iteration_per_update == 1 or \
			engine.config.iteration_per_update == 1:
			if engine.state.iteration > 1:
				engine.optimizer.zero_grad()

		device = next(engine.model.parameters()).device

		src = (mini_batch[0][0].to(device), mini_batch[0][1]) #src
		tgt = (mini_batch[1][0].to(device), mini_batch[1][1]) #tgt

		x, y = src[0], tgt[0][:, 1:]

		word_count = int(tgt[1].sum())

		with autocast(not engine.config.off_autocast):
			y_hat = engine.model(input_ids = x, labels = tgt[0][:, :-1].contiguous())
			if engine.huggingfaceUse:
				loss = y_hat.loss
			else:
				loss = engine.loss_fn(
					y_hat['logits'].contiguous().view(-1, y_hat['logits'].size(-1)),
					y.contiguous().view(-1))


			#backward_target += loss.div(y.size(0)).div(engine.config.iteration_per_update)
			if engine.loss_div == 'word_count':
				backward_target = loss.div(word_count)
			elif engine.loss_div == 'mini_batch':
				backward_target = loss.div(y.size(0))
			else:
				backward_target = loss.div(y.size(0))
			#backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)


		if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
			engine.scaler.scale(backward_target).backward()
		else:
			backward_target.backward()
		
		p_norm = float(get_parameter_norm(engine.model.parameters()))
		g_norm = float(get_grad_norm(engine.model.parameters()))
		
		if engine.state.interation % engine.config.iteration_per_update == 0 and \
			engine.state.iteration >= 0:
			torch.nn.utils.clip_grad_norm_(
				engine.model.parameters(),
				engine.config.max_grad_norm)

			if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
				engine.scaler.step(engine.optimizer)
				engine.scaler.update()
			else:
				engine.optimizer.step()

		if not engine.huggingfaceUse:
			loss = float(loss.item() / word_count)
		else:
			loss = loss.item()
		ppl  = np.exp(loss)
		metrics = {
			'loss': loss, 'ppl': ppl, 
			'|param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
			'|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
		}

		return metrics

	@staticmethod
	def validate(engine, mini_batch):
		engine.model.eval()
		with torch.no_grad():
			device = next(engine.model.parameters()).device

			src = (mini_batch[0][0].to(device), mini_batch[0][1])
			tgt = (mini_batch[1][0].to(device), mini_batch[1][1])

			x, y = src[0], tgt[0][:, 1:]

			with autocast(not engine.config.off_autocast):
				y_hat = engine.model(input_ids = x, labels = tgt[0][:, :-1].contiguous())
				if engine.huggingfaceUse:
					loss = y_hat.loss
				else:
					loss = engine.loss_fn(
						y_hat['logits'].contiguous().view(-1, y_hat['logits'].size(-1)),
						y.contiguous().view(-1))

		word_count = int(tgt[1].sum())
		if not engine.huggingfaceUse:
			loss = float(loss.item() / word_count)
		else:
			loss = loss.item()
		ppl  = np.exp(loss)

		return {'loss': loss, 'ppl': ppl}


	@staticmethod
	def attach(train_engine, validation_engine, 
			   training_metric_names = ['loss', 'ppl', '|param|', '|g_param|'],
			   validation_metric_names = ['loss', 'ppl'],
			   verbose = VERBOSE_BATCH_WISE):
		
		# Attaching would be repaeted for serveral metrics.
		# Thus, we can reduce the repeated codes by using this function.
		def attach_running_average(engine, metric_name):
			RunningAverage(output_transform = lambda x: x[metric_name]).attach(
				engine, metric_name)

		for metric_name in training_metric_names:
			attach_running_average(train_engine, metric_name)

		if verbose >= VERBOSE_BATCH_WISE:
			pbar = ProgressBar(bar_format= None, ncols = 500)
			pbar.attach(train_engine, training_metric_names)

		if verbose >= VERBOSE_EPOCH_WISE:
			@train_engine.on(Events.EPOCH_COMPLETED)
			def print_train_logs(engine):
				avg_p_norm = engine.state.metrics['|param|']
				avg_g_norm = engine.state.metrics['|g_param|']
				avg_loss   = engine.state.metrics['loss']

				print('Epoch {} - loss: {:.4e} ppl: {:.2f} |param|: {:.2e} |g_param|: {:.2e} '.format(
					engine.state.epoch,
					avg_loss,
					np.exp(avg_loss),
					avg_p_norm,
					avg_g_norm))
				if engine.lr_scheduler is not None:
					print(engine.lr)

		for metric_name in validation_metric_names:
			attach_running_average(validation_engine, metric_name)

		if verbose >= VERBOSE_BATCH_WISE:
			pbar = ProgressBar(bar_format = None, ncols = 500)
			pbar.attach(validation_engine, validation_metric_names)

		if verbose >= VERBOSE_EPOCH_WISE:
			@validation_engine.on(Events.EPOCH_COMPLETED)
			def print_valid_logs(engine):
				avg_loss = engine.state.metrics['loss']

				print('>> Validation - Loss: {:.4e}, Ppl: {:.2f} best_loss: {:.4e} best_ppl: {:.2f}'.format(
					avg_loss,
					np.exp(avg_loss),
					engine.best_loss if engine.best_loss <= 1e+3 else avg_loss,
					1e+3 if engine.best_loss >= 1e+3 else np.exp(avg_loss)))

	@staticmethod
	def record_score(train_engine, validation_engine,
		training_metric_names = ['loss', 'ppl', '|param|', '|g_param|'],
		validation_metric_names = ['loss', 'ppl'], record_lr = True):
		train_scores = train_engine.state.metrics
		valid_scores = validation_engine.state.metrics

		for i, name in enumerate(training_metric_names):
			train_engine.scores_list[i].append(train_scores[name])
			if record_lr:
				train_engine.scores_list.extend([train_engine.lr])
		for i, name in enumerate(validation_metric_names):
			validation_engine.scores_list[i].append(valid_scores[name])

	@staticmethod
	def save_record_file(train_engine, validation_engine, config, record_lr = True):
		record_file = config.record_file
		start_time  = config.nowtime
		start_epoch = config.init_epoch
		train_epoch = config.n_epochs

		if train_engine.state.epoch == config.n_epochs: 
			results = {
				'train_loss': train_engine.scores_list[0],
				'train_ppl' : train_engine.scores_list[1],
				'|param|'	: train_engine.scores_list[2],
				'|g_param|' : train_engine.scores_list[3],
				'valid_loss': validation_engine.scores_list[0],
				'valid_ppl' : validation_engine.scores_list[1]
				}
			if train_engine.record_lr:
				results.update({'train_lr':train_engine.scores_list[4]})
			now = datetime.datetime.now().strftime("%m.%d.%H")
			filepath, filename = os.path.split(record_file)
			filename, ext = os.path.splitext(filename)
			filename = os.path.join(
				filepath, 
				f'{filename}-{start_time}.e{start_epoch}~{now}.e{train_epoch}.{ext}')
			with open(filename, 'w') as f:
				f.write(str(results))

	@staticmethod
	def resume_training(engine, resume_epoch):
		engine.state.interation = (resume_epoch - 1) * len(engine.state.dataloader)
		engine.state.epoch = (resume_epoch - 1)

	@staticmethod
	def check_best(engine):
		loss = float(engine.state.metrics['loss'])
		if loss <= engine.best_loss:
			engine.best_loss = loss

	@staticmethod
	def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
		avg_train_loss = train_engine.state.metrics['loss']
		avg_valid_loss = engine.state.metrics['loss']
		model_fn = config.model_fn.split('.')
		model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
									'%.2f-%.2f' % (avg_train_loss,
												   np.exp(avg_train_loss)),
									'%.2f-%.2f' % (avg_valid_loss,
												   np.exp(avg_valid_loss))] + [model_fn[-1]]
		model_fn = '.'.join(model_fn)

		if train_engine.state.epoch == config.n_epochs or \
		train_engine.state.epoch % config.save_epoch == 1:
			torch.save(
				{
					'model': train_engine.model.state_dict(),
					'opt': train_engine.optimizer.state_dict(),
					'config': config,
					'src_vocab': src_vocab,
					'tgt_vocab': tgt_vocab,
				}, model_fn
			)


class SingleTrainer():
	def __init__(self, target_engine_class, config):
		self.target_engine_class = target_engine_class
		self.config = config

	def train(self, model, loss_fn, optimizer,
			  train_loader, valid_loader, src_vocab, tgt_vocab,
			  n_epochs, lr_scheduler = None, huggingfaceUse = False):
		
		optimizer.zero_grad()
		
		train_engine = self.target_engine_class(
			self.target_engine_class.train,
			model, loss_fn, optimizer, lr_scheduler, 
			config = self.config, huggingfaceUse = huggingfaceUse)
		
		validation_engine = self.target_engine_class(
			self.target_engine_class.validate,
			model, loss_fn, optimizer = None, lr_scheduler = None,
			config = self.config, huggingfaceUse = huggingfaceUse)

		self.target_engine_class.attach(
			train_engine, validation_engine, verbose = self.config.verbose)

		def run_validation(engine, validation_engine, valid_loader):
			validation_engine.run(valid_loader, max_epochs=1)
		
		train_engine.add_event_handler(
			Events.EPOCH_COMPLETED,
			run_validation,
			validation_engine,
			valid_loader)

		def run_lr_scheduler(engine):
			if engine.lr_scheduler is not None and \
			engine.state.iteration % self.config.iteration_per_update == 0 and \
			engine.state.iteration > 1:
				engine.lr_scheduler.step()
				engine.lr = engine.lr_scheduler.get_last_lr()

		if self.config.scheduler == 'InverseSqrtLR':
			lr_scheduler_update_unit = Events.ITERATION_COMPLETED
		else:
			lr_scheduler_update_unit = Events.EPOCH_COMPLETED
		train_engine.add_event_handler(
			lr_scheduler_update_unit,
			run_lr_scheduler)

		train_engine.add_event_handler(
			Events.STARTED,
			self.target_engine_class.resume_training,
			self.config.init_epoch)

		validation_engine.add_event_handler(
			Events.EPOCH_COMPLETED, self.target_engine_class.check_best)

		validation_engine.add_event_handler(
			Events.EPOCH_COMPLETED, self.target_engine_class.record_score,
			train_engine = train_engine, validation_engine = validation_engine)

		validation_engine.add_event_handler(
			Events.COMPLETED, self.target_engine_class.save_record_file,
			train_engine = train_engine, validation_engine = validation_engine, 
			config = self.config)
		"""
		validation_engine.add_event_handler(
			Events.EPOCH_COMPLETED, self.target_engine_class.save_model,
			train_engine, self.config, src_vocab, tgt_vocab)
		"""
		validation_engine.add_event_handler(
			Events.COMPLETED, self.target_engine_class.save_model,
			train_engine, self.config, src_vocab, tgt_vocab)


		train_engine.run(train_loader, max_epochs = n_epochs)
		return model


		






