import _pickle as cPickle
import multiprocessing
import os
import sys
import numpy as np
from joblib import Parallel, delayed
import idnns.networks.network as nn
from idnns.information import information_process  as inn
from idnns.plots import plot_figures as plt_fig
from idnns.networks import network_paramters as netp
from idnns.networks.utils import load_data
from tqdm import tqdm

# from idnns.network import utils
# import idnns.plots.plot_gradients as plt_grads
NUM_CORES = multiprocessing.cpu_count()


class informationNetwork():
	"""A class that store the network, train it and calc it's information (can be several of networks) """

	def __init__(self, rand_int=0, num_of_samples=None, args=None):
		if args == None:
			args = netp.get_default_parser(num_of_samples)
		self.cov_net = args.cov_net
		self.calc_information = args.calc_information
		self.run_in_parallel = args.run_in_parallel
		self.num_ephocs = args.num_ephocs
		self.learning_rate = args.learning_rate
		self.batch_size = args.batch_size
		self.activation_function = args.activation_function
		self.interval_accuracy_display = args.interval_accuracy_display
		self.save_grads = args.save_grads
		self.num_of_repeats = args.num_of_repeats
		self.calc_information_last = args.calc_information_last
		self.num_of_bins = args.num_of_bins
		self.interval_information_display = args.interval_information_display
		self.save_ws = args.save_ws
		self.name = args.data_dir + args.data_name
		# The arch of the networks
		self.layers_sizes = netp.select_network_arch(args.net_type)
		# The percents of the train data samples
		self.train_samples = np.linspace(1, 100, 199)[[[x * 2 - 2 for x in index] for index in args.inds]]
		# self.train_samples = np.linspace(1, 100, 199)[[[x * 2 - 2 for x in index] for index in range(0, 11416)]]
		print(f'train_samples: {len(self.train_samples)}, self.layers_sizes: {len(self.layers_sizes)}')
		# The indexs that we want to calculate the information for them in logspace interval
		self.epochs_indexes = np.unique(
			np.logspace(np.log2(args.start_samples), np.log2(args.num_ephocs), args.num_of_samples, dtype=int,
			            base=2)) - 1
		max_size = np.max([len(layers_size) for layers_size in self.layers_sizes])
		# load data
		self.data_sets = load_data(self.name, args.random_labels)
		# create arrays for saving the data
		self.ws, self.grads, self.information, self.models, self.names, self.networks, self.weights = [
			[[[[None] for k in range(len(self.train_samples))] for j in range(len(self.layers_sizes))]
			 for i in range(self.num_of_repeats)] for _ in range(7)]

		self.loss_train, self.loss_test, self.test_error, self.train_error, self.l1_norms, self.l2_norms = \
			[np.zeros((self.num_of_repeats, len(self.layers_sizes), len(self.train_samples), len(self.epochs_indexes)))
			 for _ in range(6)]

		params = {'sampleLen': len(self.train_samples),
		          'nDistSmpls': args.nDistSmpls,
		          'layerSizes': ",".join(str(i) for i in self.layers_sizes[0]), 'nEpoch': args.num_ephocs, 'batch': args.batch_size,
		          'nRepeats': args.num_of_repeats, 'nEpochInds': len(self.epochs_indexes),
		          'LastEpochsInds': self.epochs_indexes[-1], 'DataName': args.data_name,
		          'lr': args.learning_rate}

		self.name_to_save = args.name + "_" + "_".join([str(i) + '=' + str(params[i]) for i in params])
		params['train_samples'], params['CPUs'], params[
			'directory'], params['epochsInds'] = self.train_samples, NUM_CORES, self.name_to_save, self.epochs_indexes
		self.params = params
		self.rand_int = rand_int
		# If we trained already the network
		self.traind_network = False

	def save_data(self, parent_dir='jobs/', file_to_save='data.pickle'):
		"""Save the data to the file """
		directory = '{0}/{1}{2}/'.format(os.getcwd(), parent_dir, self.params['directory'])

		data = {'information': self.information,
		        'test_error': self.test_error, 'train_error': self.train_error, 'var_grad_val': self.grads,
		        'loss_test': self.loss_test, 'loss_train': self.loss_train, 'params': self.params
			, 'l1_norms': self.l1_norms, 'weights': self.weights, 'ws': self.ws}

		if not os.path.exists(directory):
			os.makedirs(directory)
		self.dir_saved = directory
		with open(self.dir_saved + file_to_save, 'wb') as f:
			cPickle.dump(data, f, protocol=2)

	def save_data_calc_only(self, parent_dir='jobs/', file_to_save='data.pickle'):
		"""Save the data to the file """
		directory = '{0}/{1}{2}/'.format(os.getcwd(), parent_dir, self.params['directory'])

		data = {'information': self.information, 'weights': self.weights, 'params': self.params, 'ws': self.ws}

		if not os.path.exists(directory):
			os.makedirs(directory)
		self.dir_saved = directory
		with open(self.dir_saved + file_to_save, 'wb') as f:
			cPickle.dump(data, f, protocol=2)

	def load_data(self, ):
		tar_dir = os.path.join('../output', 'data-%s' % 'pointnet-hsd-snn-s128-k8')
		Xs_prime, Xs_train, Ys, Ys_onehot, Zs, Logits = [], [], [], [], [], []
		EPOCHS = self.num_ephocs
		for epoch in tqdm(range(1, EPOCHS+1)): # for epochs!
			x_prime = np.load(os.path.join(tar_dir, 'x_prime-%d.npy' % epoch))          #[:600]
			x_train = np.load(os.path.join(tar_dir, 'x_train-%d.npy' % epoch))          #[:600]
			logit1 = np.load(os.path.join(tar_dir, 'logit-1-%d.npy' % epoch))           #[:600]
			logit2 = np.load(os.path.join(tar_dir, 'logit-2-%d.npy' % epoch))           #[:600]
			logit3 = np.load(os.path.join(tar_dir, 'logit-3-%d.npy' % epoch))           #[:600]
			feat1 = np.load(os.path.join(tar_dir, 'feat-1-%d.npy' % epoch))             #[:600]
			feat2 = np.load(os.path.join(tar_dir, 'feat-2-%d.npy' % epoch))             #[:600]
			feat3 = np.load(os.path.join(tar_dir, 'feat-3-%d.npy' % epoch))             #[:600]
			labelonehot = np.load(os.path.join(tar_dir, 'labelonehot-%d.npy' % epoch))  #[:600]
			label = np.load(os.path.join(tar_dir, 'label-%d.npy' % epoch))              #[:600]
			Xs_prime.append(x_prime) #.reshape(len(X_prime), -1)
			Xs_train.append(x_train.reshape(len(x_train), -1)) #
			Ys_onehot.append(labelonehot)
			Ys.append(label)
			Zs.append([feat1, feat2, feat3])
			Logits.append([logit1, logit2, logit3])
			# Zs.append([digitized_feat1, digitized_feat2, digitized_feat3])
			# Logits.append([digitized_logit1, digitized_logit2, digitized_logit3])
			# Zs.append([digitized_feat2])
			# Logits.append([digitized_logit2])
		data = np.array(Xs_train[0]) # E, B, N, 3
		print(data.shape)
		label = Ys_onehot[0] # E, B
		print(label.shape)
		ws = Zs # E, B, C
		return {'data': data, 'label': label, 'ws': Zs}


	def run_network(self):
		"""Train and calculated the network's information"""
		if self.run_in_parallel:
			results = Parallel(n_jobs=NUM_CORES)(delayed(nn.train_network)
			                                     (self.layers_sizes[j],
			                                      self.num_ephocs, self.learning_rate, self.batch_size,
			                                      self.epochs_indexes, self.save_grads, self.data_sets,
			                                      self.activation_function,
			                                      self.train_samples, self.interval_accuracy_display,
			                                      self.calc_information,
			                                      self.calc_information_last, self.num_of_bins,
			                                      self.interval_information_display, self.save_ws, self.rand_int,
			                                      self.cov_net)
			                                     for i in range(len(self.train_samples)) for j in
			                                     range(len(self.layers_sizes)) for k in range(self.num_of_repeats))

		else:
			results = [nn.train_and_calc_inf_network(i, j, k,
			                                         self.layers_sizes[j],
			                                         self.num_ephocs, self.learning_rate, self.batch_size,
			                                         self.epochs_indexes, self.save_grads, self.data_sets,
			                                         self.activation_function,
			                                         self.train_samples, self.interval_accuracy_display,
			                                         self.calc_information,
			                                         self.calc_information_last, self.num_of_bins,
			                                         self.interval_information_display,
			                                         self.save_ws, self.rand_int, self.cov_net)
			           for i in range(len(self.train_samples)) for j in range(len(self.layers_sizes)) for k in
			           range(self.num_of_repeats)]

		# Extract all the measures and orgainze it
		for i in range(len(self.train_samples)):
			for j in range(len(self.layers_sizes)):
				for k in range(self.num_of_repeats):
					index = i * len(self.layers_sizes) * self.num_of_repeats + j * self.num_of_repeats + k
					print('index', index)
					current_network = results[index]
					print('ws', len(current_network['ws']), len(current_network['ws'][0])) # list
					# t = current_network['ws'][0]
					# for tt in t:
					# 	print(len(tt)) # 4096
					# print('information', current_network['information'].shape, current_network['information'][0, 0, :]) # np array
					# information: 1, epoch, 6 layers
					# print('weights', current_network['weights'], type(current_network['weights'])) # always int 0
					self.networks[k][j][i] = current_network
					self.ws[k][j][i] = current_network['ws']
					self.weights[k][j][i] = current_network['weights']
					self.information[k][j][i] = current_network['information']
					self.grads[k][i][i] = current_network['gradients']
					self.test_error[k, j, i, :] = current_network['test_prediction']
					self.train_error[k, j, i, :] = current_network['train_prediction']
					self.loss_test[k, j, i, :] = current_network['loss_test']
					self.loss_train[k, j, i, :] = current_network['loss_train']
		# print(f'self.ws, weights: {self.ws.shape, self.weights.shape}')
		self.traind_network = True

	def run_calc_only(self):
		"""Train and calculated the network's information"""
		if self.run_in_parallel:
			results = Parallel(n_jobs=NUM_CORES)(delayed(nn.train_network)
			                                     (self.layers_sizes[j],
			                                      self.num_ephocs, self.learning_rate, self.batch_size,
			                                      self.epochs_indexes, self.save_grads, self.data_sets,
			                                      self.activation_function,
			                                      self.train_samples, self.interval_accuracy_display,
			                                      self.calc_information,
			                                      self.calc_information_last, self.num_of_bins,
			                                      self.interval_information_display, self.save_ws, self.rand_int,
			                                      self.cov_net)
			                                     for i in range(len(self.train_samples)) for j in
			                                     range(len(self.layers_sizes)) for k in range(self.num_of_repeats))

		else:
			results = [nn.calc_only(self.load_data, i, j, k,
			                                         self.layers_sizes[j],
			                                         self.num_ephocs, self.learning_rate, self.batch_size,
			                                         self.epochs_indexes, self.save_grads, self.data_sets,
			                                         self.activation_function,
			                                         self.train_samples, self.interval_accuracy_display,
			                                         self.calc_information,
			                                         self.calc_information_last, self.num_of_bins,
			                                         self.interval_information_display,
			                                         self.save_ws, self.rand_int, self.cov_net)
			           for i in range(len(self.train_samples)) for j in range(len(self.layers_sizes)) for k in
			           range(self.num_of_repeats)]

		# Extract all the measures and orgainze it
		for i in range(len(self.train_samples)):
			for j in range(len(self.layers_sizes)):
				for k in range(self.num_of_repeats):
					index = i * len(self.layers_sizes) * self.num_of_repeats + j * self.num_of_repeats + k
					current_network = results[index]
					self.networks[k][j][i] = current_network
					self.ws[k][j][i] = current_network['ws']
					self.weights[k][j][i] = current_network['weights']
					self.information[k][j][i] = current_network['information']
					# self.grads[k][i][i] = current_network['gradients']
					# self.test_error[k, j, i, :] = current_network['test_prediction']
					# self.train_error[k, j, i, :] = current_network['train_prediction']
					# self.loss_test[k, j, i, :] = current_network['loss_test']
					# self.loss_train[k, j, i, :] = current_network['loss_train']
		# print(f'self.ws, weights: {self.ws.shape, self.weights.shape}')
		self.traind_network = True

	def print_information(self):
		"""Print the networks params"""
		for val in self.params:
			if val != 'epochsInds':
				print (val, self.params[val])

	def calc_information(self):
		"""Calculate the infomration of the network for all the epochs - only valid if we save the activation values and trained the network"""
		if self.traind_network and self.save_ws:
			self.information = np.array(
				[inn.get_information(self.ws[k][j][i], self.data_sets.data, self.data_sets.labels,
				                     self.args.num_of_bins, self.args.interval_information_display, self.epochs_indexes)
				 for i in range(len(self.train_samples)) for j in
				 range(len(self.layers_sizes)) for k in range(self.args.num_of_repeats)])
		else:
			print ('Cant calculate the infomration of the networks!!!')

	def calc_information_last(self):
		"""Calculate the information of the last epoch"""
		if self.traind_network and self.save_ws:
			return np.array([inn.get_information([self.ws[k][j][i][-1]], self.data_sets.data, self.data_sets.labels,
			                                     self.args.num_of_bins, self.args.interval_information_display,
			                                     self.epochs_indexes)
			                 for i in range(len(self.train_samples)) for j in
			                 range(len(self.layers_sizes)) for k in range(self.args.num_of_repeats)])

	def plot_network(self):
		str_names = [[self.dir_saved]]
		mode = 2
		save_name = 'figure'
		plt_fig.plot_figures(str_names, mode, save_name)

	def plot_network_calc_only(self):
		str_names = [[self.dir_saved]]
		mode = 2
		save_name = 'figure'
		plt_fig.plot_figures_calc_only(str_names, mode, save_name)
