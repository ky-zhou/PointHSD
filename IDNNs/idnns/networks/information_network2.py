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
# from idnns.networks.utils import load_data
from tqdm import tqdm
from sklearn.decomposition import PCA

# from idnns.network import utils
# import idnns.plots.plot_gradients as plt_grads
NUM_CORES = multiprocessing.cpu_count()


class informationNetwork():
	"""A class that store the network, train it and calc it's information (can be several of networks) """

	def __init__(self, rand_int=0, num_of_samples=None, args=None):
		if args == None:
			args = netp.get_default_parser(num_of_samples)
		self.cov_net = args.cov_net
		self.if_calc_information = args.calc_information
		self.run_in_parallel = args.run_in_parallel
		self.num_ephocs = args.num_ephocs
		self.learning_rate = args.learning_rate
		self.batch_size = args.batch_size
		self.activation_function = args.activation_function
		self.interval_accuracy_display = args.interval_accuracy_display
		self.save_grads = args.save_grads
		self.num_of_repeats = args.num_of_repeats
		self.if_calc_information_last = args.calc_information_last
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
		# self.epochs_indexes = np.unique(
		# 	np.logspace(np.log2(args.start_samples), np.log2(args.num_ephocs), args.num_of_samples, dtype=int,
		# 	            base=2)) - 1
		self.epochs_indexes = np.arange(self.num_ephocs)
		print(2, self.epochs_indexes)
		max_size = np.max([len(layers_size) for layers_size in self.layers_sizes])
		# create arrays for saving the data
		self.ws, self.information, self.weights = [
			[[[[None] for k in range(len(self.train_samples))] for j in range(len(self.layers_sizes))]
			 for i in range(self.num_of_repeats)] for _ in range(3)]

		params = {'sampleLen': len(self.train_samples),
		          'nDistSmpls': args.nDistSmpls,
		          'layerSizes': ",".join(str(i) for i in self.layers_sizes[0]), 'nEpoch': args.num_ephocs,
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
		tar_dir = os.path.join('../output', 'data-%s' % 'pointnet-hsd-snn-s128-k8_worked')
		Xs_prime, Xs_train, Ys, Ys_onehot, Zs, Logits = [], [], [], [], [], []
		EPOCHS = self.num_ephocs
		num_bins = 100
		digitize = False
		pca = PCA(1)
		for epoch in tqdm(range(1, EPOCHS+1)): # for epochs!
			if epoch == 1:
				x_prime = np.load(os.path.join(tar_dir, 'x_prime-%d.npy' % epoch))          [:400]
				x_train = np.load(os.path.join(tar_dir, 'x_train-%d.npy' % epoch))          [:400]
				labelonehot = np.load(os.path.join(tar_dir, 'labelonehot-%d.npy' % epoch))  [:400]
				label = np.load(os.path.join(tar_dir, 'label-%d.npy' % epoch))              [:400]
				Xs_prime.append(x_prime.reshape(len(x_prime), -1)) #.reshape(len(X_prime), -1)
				Xs_train.append(x_train.reshape(len(x_prime), -1)) #.reshape(len(X_prime), -1)
				Ys_onehot.append(labelonehot)
				Ys.append(label)
			logit1 = np.load(os.path.join(tar_dir, 'logit-1-%d.npy' % epoch))           	[:400]
			logit2 = np.load(os.path.join(tar_dir, 'logit-2-%d.npy' % epoch))           	[:400]
			logit3 = np.load(os.path.join(tar_dir, 'logit-3-%d.npy' % epoch))           	[:400]
			feat1 = np.load(os.path.join(tar_dir, 'xyz-1-%d.npy' % epoch))             		[:400]
			feat2 = np.load(os.path.join(tar_dir, 'xyz-2-%d.npy' % epoch))             		[:400]
			feat3 = np.load(os.path.join(tar_dir, 'xyz-3-%d.npy' % epoch))             		[:400]
			if digitize:
				size = len(x_train)
				new_x_train, new_x_prime = [], []
				for l in range(size):
					x = pca.fit_transform(x_train[l])
					new_x_train.append(x)
					new_x_prime.append(x)
					# print(x.shape)
				new_x_train = np.array(new_x_train).squeeze()
				new_x_prime = np.array(new_x_prime).squeeze()
				bins_train = np.linspace(new_x_train.min(), new_x_train.max(), num_bins)
				digitized_train = np.digitize(new_x_train, bins_train)
				digitized_prime = np.digitize(new_x_prime, bins_train)

				# assert 1==2
				bins_logit = np.linspace(min(logit1.min(), logit2.min(), logit3.min()), max(logit1.max(), logit2.max(), logit3.max()), num_bins)
				digitized_logit1 = np.digitize(logit1, bins_logit)
				digitized_logit2 = np.digitize(logit2, bins_logit)
				digitized_logit3 = np.digitize(logit3, bins_logit)
				bins_feat = np.linspace(min(feat1.min(), feat2.min(), feat3.min()), max(feat1.max(), feat2.max(), feat3.max()), num_bins)
				digitized_feat1 = np.digitize(feat1, bins_feat)
				digitized_feat2 = np.digitize(feat2, bins_feat)
				digitized_feat3 = np.digitize(feat3, bins_feat)
				length = len(x_prime)
				# Zs.append([digitized_feat1.reshape(length, -1), digitized_feat2.reshape(length, -1), digitized_feat3.reshape(length, -1)])
				# Logits.append([digitized_logit1.reshape(length, -1), digitized_logit2.reshape(length, -1), digitized_logit3.reshape(length, -1)])
				Zs.append([digitized_feat1, digitized_feat2, digitized_feat3])
				Logits.append([digitized_logit1, digitized_logit2, digitized_logit3])
			# print(digitized_feat1.shape)
			else:
				# Xs_prime.append(digitized_prime) #.reshape(len(X_prime), -1)
				# Xs_train.append(digitized_train) #.reshape(len(X_prime), -1)
				Zs.append([feat1.reshape(len(x_train), -1), feat2.reshape(len(x_train), -1), feat3.reshape(len(x_train), -1)])
				Logits.append([logit1, logit2, logit3])
				# Zs.append([feat3.reshape(len(x_train), -1)])
				# Logits.append([logit3])
		data = np.array(Xs_train[0]) # E, B, N, 3
		print(data.shape)
		label = Ys_onehot[0] # E, B
		print(label.shape)
		ws = Zs # E, B, C
		return {'data': data, 'label': label, 'ws': Logits}

	def run_calc_only(self):
		"""Train and calculated the network's information"""
		if self.run_in_parallel:
			results = Parallel(n_jobs=NUM_CORES)(delayed(nn.train_network)
			                                     (self.layers_sizes[j],
			                                      self.num_ephocs, self.epochs_indexes,
			                                      self.activation_function, self.train_samples, 
			                                      self.if_calc_information,
			                                      self.if_calc_information_last, self.num_of_bins,
			                                      self.interval_information_display, self.save_ws, self.rand_int,
			                                      )
			                                     for i in range(len(self.train_samples)) for j in
			                                     range(len(self.layers_sizes)) for k in range(self.num_of_repeats))

		else:
			results = [nn.calc_only(self.load_data, i, j, k,
			                                         self.layers_sizes[j],
			                                         self.num_ephocs, self.epochs_indexes,
			                                         self.activation_function, self.train_samples, 
			                                         self.if_calc_information,
			                                         self.if_calc_information_last, self.num_of_bins,
			                                         self.interval_information_display,
			                                         self.save_ws, self.rand_int)
			           for i in range(len(self.train_samples)) for j in range(len(self.layers_sizes)) for k in
			           range(self.num_of_repeats)]

		# Extract all the measures and orgainze it
		for i in range(len(self.train_samples)):
			for j in range(len(self.layers_sizes)):
				for k in range(self.num_of_repeats):
					index = i * len(self.layers_sizes) * self.num_of_repeats + j * self.num_of_repeats + k
					print('index', index)
					current_network = results[index]
					print('ws', len(current_network['ws']), len(current_network['ws'][0])) # Eopch*n_layer
					# t = current_network['ws'][0]
					# for tt in t:
					# 	print(len(tt)) # 4096
					print('information', current_network['information'].shape, current_network['information']) # np array
					# information: 1, epoch, 6 layers
					# print('weights', current_network['weights'], type(current_network['weights'])) # always int 0
					# current_network = results[index]
					# self.networks[k][j][i] = current_network
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

	def plot_network_calc_only(self):
		str_names = [[self.dir_saved]]
		mode = 9
		save_name = 'infop'
		plt_fig.plot_figures_calc_only(str_names, mode, save_name)
		# plt_fig.plot_figures_calc_only_delta(str_names, mode, save_name+'_delta')
