# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ train_utils.py ]
#   Synopsis     [ helper functions for train.py ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import time
import pickle
from datetime import datetime
#---------------------------#
import tensorflow as tf
#---------------------------#
from config import get_config
args = get_config()


"""
	Convert raw audio to spectrogram
"""
def t_to_f(x, X_mean, X_std):
	x = x[:, :, 0]
	X = tf.contrib.signal.stft(x, 256, 128, pad_end=True)
	X = X[:, :, :-1]

	X_mag = tf.abs(X)
	X_lmag = tf.log(X_mag + args._LOG_EPS)
	X_norm = (X_lmag - X_mean[:-1]) / X_std[:-1]
	X_norm /= args._CLIP_NSTD
	X_norm = tf.clip_by_value(X_norm, -1., 1.)
	X_norm = tf.expand_dims(X_norm, axis=3)

	X_norm = tf.stop_gradient(X_norm)

	return X_norm


"""
	Griffin-Lim
"""
def invert_spectra_griffin_lim(X_mag, nfft, nhop, ngl):
	X = tf.complex(X_mag, tf.zeros_like(X_mag))

	def b(i, X_best):
		x = tf.contrib.signal.inverse_stft(X_best, nfft, nhop)
		X_est = tf.contrib.signal.stft(x, nfft, nhop)
		phase = X_est / tf.cast(tf.maximum(1e-8, tf.abs(X_est)), tf.complex64)
		X_best = X * phase
		return i + 1, X_best

	i = tf.constant(0)
	c = lambda i, _: tf.less(i, ngl)
	_, X = tf.while_loop(c, b, [i, X], back_prop=False)

	x = tf.contrib.signal.inverse_stft(X, nfft, nhop)
	x = x[:, :args._WINDOW_LEN]

	return x


"""
	Estimate raw audio for spectrogram
"""
def f_to_t(X_norm, X_mean, X_std, ngl=16):
	X_norm = X_norm[:, :, :, 0]
	X_norm = tf.pad(X_norm, [[0,0], [0,0], [0,1]])
	X_norm *= args._CLIP_NSTD
	X_lmag = (X_norm * X_std) + X_mean
	X_mag = tf.exp(X_lmag)

	x = invert_spectra_griffin_lim(X_mag, 256, 128, ngl)
	x = tf.reshape(x, [-1, args._WINDOW_LEN, 1])

	return x


"""
	Render normalized spectrogram as uint8 image
"""
def f_to_img(X_norm):
	X_uint8 = X_norm + 1.
	X_uint8 *= 128.
	X_uint8 = tf.clip_by_value(X_uint8, 0., 255.)
	X_uint8 = tf.cast(X_uint8, tf.uint8)

	X_uint8 = tf.map_fn(lambda x: tf.image.rot90(x, 1), X_uint8)

	return X_uint8


"""
	for printing training logs during MonitoredTrainingSession
"""
class tf_train_LoggerHook(tf.train.SessionRunHook):

	def __init__(self, args, losses):
		self.train_dir = args.train_dir
		self.display_step = args.train_display_step
		self.batch_size = args.train_batch_size
		self.save_log_step = args.train_save_log_step
		self.D_loss = losses[0]
		self.G_loss = losses[1]
		try: assert len(losses) <= 2
		except: raise NotImplementedError()
	
	def begin(self):
		self._step = -1
		self._start_time = time.time()
		try:
			self.D_log = pickle.load(open(os.path.join(self.train_dir, 'd_log.pkl'), 'rb'))
			self.G_log = pickle.load(open(os.path.join(self.train_dir, 'g_log.pkl'), 'rb'))
			print('Loading and logging into pre-existing training log...')
			print('-' * 80)
		except:
			self.D_log = []
			self.G_log = []
			print('Creating new training log...')

	def before_run(self, run_context):
		self._step += 1
		# returns the tensor or op in [] for inspection during the training session
		return tf.train.SessionRunArgs([self.D_loss, self.G_loss])

	def after_run(self, run_context, run_values):
		if self._step % self.display_step == 0:

			# calculate time
			current_time = time.time()
			duration = current_time - self._start_time
			self._start_time = current_time

			# reuslts store the returned values from before_run(), returns a list if given a list input
			D_loss = run_values.results[0]
			G_loss = run_values.results[1]

			# compute information
			examples_per_sec = self.display_step * self.batch_size / duration
			sec_per_batch = float(duration / self.display_step)

			# print
			format_str = ('%s: step %d, D_loss = %.2f, G_loss = %.2f, (%.1f examples/sec; %.3f sec/batch)')
			print(format_str % (datetime.now(), self._step, D_loss, G_loss, examples_per_sec, sec_per_batch))

		if self._step % self.save_log_step == 0:
			pickle.dump(self.D_log, open(os.path.join(self.train_dir, 'd_log.pkl'), 'wb'), True)
			pickle.dump(self.G_log, open(os.path.join(self.train_dir, 'g_log.pkl'), 'wb'), True)


