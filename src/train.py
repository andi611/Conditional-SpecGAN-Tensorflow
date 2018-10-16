# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ train.py ]
#   Synopsis     [ train the SpecGAN model ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import time
import pickle
from functools import reduce
#---------------------------#
import numpy as np
import tensorflow as tf
#---------------------------#
import loader
import helper
from config import get_config
from model import word_embedding, Spec_GAN_Generator, Spec_GAN_Discriminator


"""
	Trains a (conditional) SpecGAN using MonitoredTrainingSession
"""
def train(fps, args, cond=False):
	
	with tf.name_scope('loader'):
		if cond:
			#---data flow of a batch of (x, y) training data--@
			x_wav, y_label = loader.get_batch(fps, args.train_batch_size, args._WINDOW_LEN, args.data_first_window, labels=True)
			
			#---parse label from tensor to categorical---#
			y, w = loader.label_to_tensor(label=y_label, args=args, fps=fps)  # (right, wrong) labels

		else:
			x_wav = loader.get_batch(fps, args.train_batch_size, args._WINDOW_LEN, args.data_first_window)
		x = helper.t_to_f(x_wav, args.data_moments_mean, args.data_moments_std)
	
	#---word embedding---#
	if cond:
		with tf.variable_scope('word_embedding'):
			y_emb = word_embedding(word=y, vocab_size=args._VOCAB_SIZE, embedding_dim=args.SpecGAN_word_embedding_dim, train=False)
		with tf.variable_scope('word_embedding', reuse=True):
			w_emb = word_embedding(word=w, vocab_size=args._VOCAB_SIZE, embedding_dim=args.SpecGAN_word_embedding_dim, train=False)		
	
	# Make z vector
	if args.SpecGAN_prior_noise == 'uniform':
		z = tf.random_uniform([args.train_batch_size, args._D_Z], minval=-1., maxval=1., dtype=tf.float32)
	elif args.SpecGAN_prior_noise == 'normal':
		z = tf.random_normal([args.train_batch_size, args._D_Z], mean=0., stddev=1., dtype=tf.float32)
	else:
		raise NotImplementedError()


	# Make generator
	with tf.variable_scope('G'):
		if cond: 
			G_z = Spec_GAN_Generator(z, word=y_emb, train=True, cond=True, **args.SpecGAN_g_kwargs)
		else:
			G_z = Spec_GAN_Generator(z, train=True, **args.SpecGAN_g_kwargs)
	G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')


	# Print G summary
	print('-' * 80)
	print('####### Generator vars #######')
	nparams = 0
	for v in G_vars:
		v_shape = v.get_shape().as_list()
		v_n = reduce(lambda x, y: x * y, v_shape)
		nparams += v_n
		print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
	print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))


	# Make real discriminator
	with tf.name_scope('D_x'), tf.variable_scope('D'):
		if cond:
			D_x = Spec_GAN_Discriminator(x, word=y_emb, cond=True, **args.SpecGAN_d_kwargs)
		else:
			D_x = Spec_GAN_Discriminator(x, **args.SpecGAN_d_kwargs)
	D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')


	# Print D summary
	print('-' * 80)
	print('####### Discriminator vars #######')
	nparams = 0
	for v in D_vars:
		v_shape = v.get_shape().as_list()
		v_n = reduce(lambda x, y: x * y, v_shape)
		nparams += v_n
		print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
	print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
	print('-' * 80)


	# Make fake discriminator
	with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
		if cond:
			D_G_z = Spec_GAN_Discriminator(G_z, word=y_emb, cond=True, **args.SpecGAN_d_kwargs)
		else:
			D_G_z = Spec_GAN_Discriminator(G_z, **args.SpecGAN_d_kwargs)

	# Make mismatch discriminator
	with tf.name_scope('D_G_w'), tf.variable_scope('D', reuse=True):
		if cond:
			D_x_w = Spec_GAN_Discriminator(x, word=w_emb, cond=True, **args.SpecGAN_d_kwargs)


	# Create loss
	D_clip_weights = None
	if args.SpecGAN_loss == 'dcgan':
		if cond: raise NotImplementedError()
		fake = tf.zeros([args.train_batch_size], dtype=tf.float32)
		real = tf.ones([args.train_batch_size], dtype=tf.float32)

		G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=D_G_z,
			labels=real
		))

		D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=D_G_z,
			labels=fake
		))
		D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=D_x,
			labels=real
		))

		D_loss /= 2.
	elif args.SpecGAN_loss == 'lsgan':
		if cond: raise NotImplementedError()
		G_loss = tf.reduce_mean((D_G_z - 1.) ** 2)
		D_loss = tf.reduce_mean((D_x - 1.) ** 2)
		D_loss += tf.reduce_mean(D_G_z ** 2)
		D_loss /= 2.
	elif args.SpecGAN_loss == 'wgan':
		if cond: raise NotImplementedError()
		G_loss = -tf.reduce_mean(D_G_z)
		D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

		with tf.name_scope('D_clip_weights'):
			clip_ops = []
			for var in D_vars:
				clip_bounds = [-.01, .01]
				clip_ops.append(
					tf.assign(
						var,
						tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
					)
				)
			D_clip_weights = tf.group(*clip_ops)
	elif args.SpecGAN_loss == 'wgan-gp':
		G_loss = - tf.reduce_mean(D_G_z) # - D fake
		if cond:
			D_loss = - (tf.reduce_mean(D_x) - tf.reduce_mean(D_G_z))
			D_loss += - (tf.reduce_mean(D_x) - tf.reduce_mean(D_x_w))
		else:
			D_loss = - (tf.reduce_mean(D_x) - tf.reduce_mean(D_G_z)) # min (D real - D fake) => D fake - D real

		alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1, 1], minval=0., maxval=1.)
		differences = G_z - x
		interpolates = x + (alpha * differences)
		with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
			D_interp = Spec_GAN_Discriminator(interpolates, **args.SpecGAN_d_kwargs)

		LAMBDA = 10
		gradients = tf.gradients(D_interp, [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
		gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
		D_loss += LAMBDA * gradient_penalty
	else:
		raise NotImplementedError()




	# Create (recommended) optimizer
	if args.SpecGAN_loss == 'dcgan':
		G_opt = tf.train.AdamOptimizer(
				learning_rate=2e-4,
				beta1=0.5)
		D_opt = tf.train.AdamOptimizer(
				learning_rate=2e-4,
				beta1=0.5)
	elif args.SpecGAN_loss == 'lsgan':
		G_opt = tf.train.RMSPropOptimizer(
				learning_rate=1e-4)
		D_opt = tf.train.RMSPropOptimizer(
				learning_rate=1e-4)
	elif args.SpecGAN_loss == 'wgan':
		G_opt = tf.train.RMSPropOptimizer(
				learning_rate=5e-5)
		D_opt = tf.train.RMSPropOptimizer(
				learning_rate=5e-5)
	elif args.SpecGAN_loss == 'wgan-gp':
		G_opt = tf.train.AdamOptimizer(
				learning_rate=1e-4,
				beta1=0.5,
				beta2=0.9)
		D_opt = tf.train.AdamOptimizer(
				learning_rate=1e-4,
				beta1=0.5,
				beta2=0.9)
	else:
		raise NotImplementedError()

	# Summarize
	x_gl = helper.f_to_t(x, args.data_moments_mean, args.data_moments_std, args.SpecGAN_ngl)
	G_z_gl = helper.f_to_t(G_z, args.data_moments_mean, args.data_moments_std, args.SpecGAN_ngl)
	tf.summary.audio('x_wav', x_wav, args._FS)
	tf.summary.audio('x', x_gl, args._FS)
	tf.summary.audio('G_z', G_z_gl, args._FS)
	G_z_rms = tf.sqrt(tf.reduce_mean(tf.square(G_z_gl[:, :, 0]), axis=1))
	x_rms = tf.sqrt(tf.reduce_mean(tf.square(x_gl[:, :, 0]), axis=1))
	tf.summary.histogram('x_rms_batch', x_rms)
	tf.summary.histogram('G_z_rms_batch', G_z_rms)
	tf.summary.scalar('x_rms', tf.reduce_mean(x_rms))
	tf.summary.scalar('G_z_rms', tf.reduce_mean(G_z_rms))
	tf.summary.image('x', helper.f_to_img(x))
	tf.summary.image('G_z', helper.f_to_img(G_z))
	try:
		W_distance = tf.reduce_mean(D_x) - 2*tf.reduce_mean(D_G_z)
		tf.summary.scalar('W_distance', W_distance)
	except: pass
	tf.summary.scalar('G_loss', G_loss)
	tf.summary.scalar('D_loss', D_loss)

	# Create training ops
	G_train_op = G_opt.minimize(G_loss, var_list=G_vars, global_step=tf.train.get_or_create_global_step())
	D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

	# Global step defiend for StopAtStepHook
	global_step = tf.train.get_or_create_global_step()

	# Run training
	with tf.train.MonitoredTrainingSession(
					hooks=[tf.train.StopAtStepHook(last_step=args.train_max_step), # hook that stops training at max_step
						   tf.train.NanTensorHook(D_loss), # hook that monitors the loss, terminate if loss is NaN
						   helper.tf_train_LoggerHook(args, losses=[D_loss, G_loss, W_distance])], # user defiend log printing hook
					checkpoint_dir=args.train_dir,
					save_checkpoint_secs=args.train_save_secs,
					save_summaries_secs=args.train_summary_secs) as sess:
		while not sess.should_stop():
			# Train discriminator
			for i in range(args.SpecGAN_disc_nupdates):
				sess.run(D_train_op)

				# Enforce Lipschitz constraint for WGAN
				if D_clip_weights is not None:
					sess.run(D_clip_weights)

			# Train generator
			sess.run(G_train_op)
	

"""
	Creates and saves a MetaGraphDef for simple inference
	Tensors:
		'samp_z_n' int32 []: Sample this many latent vectors
		'samp_z' float32 [samp_z_n, 100]: Resultant latent vectors
		'w:0' float32 [None, 10]: One-hot vectors representing vocabularies
		'z:0' float32 [None, 100]: Input latent vectors
		'ngl:0' int32 []: Number of Griffin-Lim iterations for resynthesis
		'flat_pad:0' int32 []: Number of padding samples to use when flattening batch to a single audio file
		'G_z_norm:0' float32 [None, 128, 128, 1]: Generated outputs (frequency domain)
		'G_z:0' float32 [None, 16384, 1]: Generated outputs (Griffin-Lim'd to time domain)
		'G_z_norm_uint8:0' uint8 [None, 128, 128, 1]: Preview speechtrogram image
		'G_z_int16:0' int16 [None, 16384, 1]: Same as above but quantizied to 16-bit PCM samples
		'G_z_flat:0' float32 [None, 1]: Outputs flattened into single audio file
		'G_z_flat_int16:0' int16 [None, 1]: Same as above but quantized to 16-bit PCM samples
	Example usage:
		import tensorflow as tf
		tf.reset_default_graph()

		saver = tf.train.import_meta_graph('infer.meta')
		graph = tf.get_default_graph()
		sess = tf.InteractiveSession()
		saver.restore(sess, 'model.ckpt-10000')

		z_n = graph.get_tensor_by_name('samp_z_n:0')
		_z = sess.run(graph.get_tensor_by_name('samp_z:0'), {z_n: 10})

		z = graph.get_tensor_by_name('G_z:0')
		_G_z = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z})
"""
def infer(args, cond=False):
	infer_dir = os.path.join(args.train_dir, 'infer')
	if not os.path.isdir(infer_dir):
		os.makedirs(infer_dir)

	# Subgraph that generates latent vectors
	samp_z_n = tf.placeholder(tf.int32, [], name='samp_z_n')
	samp_z = tf.random_uniform([samp_z_n, args._D_Z], -1.0, 1.0, dtype=tf.float32, name='samp_z')

	# Input zo
	z = tf.placeholder(tf.float32, [None, args._D_Z], name='z')
	ngl = tf.placeholder(tf.int32, [], name='ngl')
	flat_pad = tf.placeholder(tf.int32, [], name='flat_pad')
	if cond:
		idx_word_label = tf.placeholder(tf.int32, [None], name='cond_word_label')
		tensor = loader.idx_to_categorical(idx_word_label, args)
		with tf.variable_scope('word_embedding'):
			emb = word_embedding(word=tensor, vocab_size=args._VOCAB_SIZE, embedding_dim=args.SpecGAN_word_embedding_dim, train=False)

	# Execute generator
	with tf.variable_scope('G'):
		if cond: 
			G_z_norm = Spec_GAN_Generator(z, word=emb, train=False, cond=True, **args.SpecGAN_g_kwargs)
		else:
			G_z_norm = Spec_GAN_Generator(z, train=False, **args.SpecGAN_g_kwargs)
	G_z_norm = tf.identity(G_z_norm, name='G_z_norm')
	G_z = helper.f_to_t(G_z_norm, args.data_moments_mean, args.data_moments_std, ngl)
	G_z = tf.identity(G_z, name='G_z')

	G_z_norm_uint8 = helper.f_to_img(G_z_norm)
	G_z_norm_uint8 = tf.identity(G_z_norm_uint8, name='G_z_norm_uint8')

	# Flatten batch
	nch = int(G_z.get_shape()[-1])
	G_z_padded = tf.pad(G_z, [[0, 0], [0, flat_pad], [0, 0]])
	G_z_flat = tf.reshape(G_z_padded, [-1, nch], name='G_z_flat')

	# Encode to int16
	def float_to_int16(x, name=None):
		x_int16 = x * 32767.
		x_int16 = tf.clip_by_value(x_int16, -32767., 32767.)
		x_int16 = tf.cast(x_int16, tf.int16, name=name)
		return x_int16
	G_z_int16 = float_to_int16(G_z, name='G_z_int16')
	G_z_flat_int16 = float_to_int16(G_z_flat, name='G_z_flat_int16')

	# Create saver
	G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
	w_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='word_embedding')
	global_step = tf.train.get_or_create_global_step()
	saver = tf.train.Saver(G_vars + w_vars + [global_step])

	# Export graph
	tf.train.write_graph(tf.get_default_graph(), infer_dir, 'infer.pbtxt')

	# Export MetaGraph
	infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
	tf.train.export_meta_graph(
			filename=infer_metagraph_fp,
			clear_devices=True,
			saver_def=saver.as_saver_def())

	# Reset graph (in case training afterwards)
	tf.reset_default_graph()


"""
	Generates a preview audio file every time a checkpoint is saved
"""
def preview(args):
	from scipy.io.wavfile import write as wavwrite
	from scipy.signal import freqz

	preview_dir = os.path.join(args.train_dir, 'preview')
	if not os.path.isdir(preview_dir):
		os.makedirs(preview_dir)

	# Load graph
	infer_metagraph_fp = os.path.join(args.train_dir, 'infer', 'infer.meta')
	graph = tf.get_default_graph()
	saver = tf.train.import_meta_graph(infer_metagraph_fp)

	# Generate or restore z_i and z_o
	z_fp = os.path.join(preview_dir, 'z.pkl')
	if os.path.exists(z_fp):
		with open(z_fp, 'rb') as f:
			_zs = pickle.load(f)
	else:
		# Sample z
		samp_feeds = {}
		samp_feeds[graph.get_tensor_by_name('samp_z_n:0')] = args.preview_n
		samp_fetches = {}
		samp_fetches['zs'] = graph.get_tensor_by_name('samp_z:0')
		with tf.Session() as sess:
			_samp_fetches = sess.run(samp_fetches, samp_feeds)
		_zs = _samp_fetches['zs']

		# Save z
		with open(z_fp, 'wb') as f:
			pickle.dump(_zs, f)

	# Set up graph for generating preview images
	feeds = {}
	feeds[graph.get_tensor_by_name('z:0')] = _zs
	feeds[graph.get_tensor_by_name('ngl:0')] = args.SpecGAN_ngl
	feeds[graph.get_tensor_by_name('flat_pad:0')] = args._WINDOW_LEN // 2
	fetches =  {}
	fetches['step'] = tf.train.get_or_create_global_step()
	fetches['G_z'] = graph.get_tensor_by_name('G_z:0')
	fetches['G_z_flat_int16'] = graph.get_tensor_by_name('G_z_flat_int16:0')

	# Summarize
	G_z = graph.get_tensor_by_name('G_z_flat:0')
	summaries = [
			tf.summary.audio('preview', tf.expand_dims(G_z, axis=0), args._FS, max_outputs=1)
	]
	fetches['summaries'] = tf.summary.merge(summaries)
	summary_writer = tf.summary.FileWriter(preview_dir)

	# Loop, waiting for checkpoints
	ckpt_fp = None
	while True:
		latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
		if latest_ckpt_fp != ckpt_fp:
			print('Preview: {}'.format(latest_ckpt_fp))

			with tf.Session() as sess:
				saver.restore(sess, latest_ckpt_fp)

				_fetches = sess.run(fetches, feeds)

				_step = _fetches['step']

			preview_fp = os.path.join(preview_dir, '{}.wav'.format(str(_step).zfill(8)))
			wavwrite(preview_fp, args._FS, _fetches['G_z_flat_int16'])

			summary_writer.add_summary(_fetches['summaries'], _step)

			print('Done')

			ckpt_fp = latest_ckpt_fp

		time.sleep(1)


"""
	Computes inception score every time a checkpoint is saved
"""
def incept(args):
	incept_dir = os.path.join(args.train_dir, 'incept')
	if not os.path.isdir(incept_dir):
		os.makedirs(incept_dir)

	# Load GAN graph
	gan_graph = tf.Graph()
	with gan_graph.as_default():
		infer_metagraph_fp = os.path.join(args.train_dir, 'infer', 'infer.meta')
		gan_saver = tf.train.import_meta_graph(infer_metagraph_fp)
		score_saver = tf.train.Saver(max_to_keep=1)
	gan_z = gan_graph.get_tensor_by_name('z:0')
	gan_ngl = gan_graph.get_tensor_by_name('ngl:0')
	gan_G_z = gan_graph.get_tensor_by_name('G_z:0')[:, :, 0]
	gan_step = gan_graph.get_tensor_by_name('global_step:0')

	# Load or generate latents
	z_fp = os.path.join(incept_dir, 'z.pkl')
	if os.path.exists(z_fp):
		with open(z_fp, 'rb') as f:
			_zs = pickle.load(f)
	else:
		gan_samp_z_n = gan_graph.get_tensor_by_name('samp_z_n:0')
		gan_samp_z = gan_graph.get_tensor_by_name('samp_z:0')
		with tf.Session(graph=gan_graph) as sess:
			_zs = sess.run(gan_samp_z, {gan_samp_z_n: args.incept_n})
		with open(z_fp, 'wb') as f:
			pickle.dump(_zs, f)

	# Load classifier graph
	incept_graph = tf.Graph()
	with incept_graph.as_default():
		incept_saver = tf.train.import_meta_graph(args.incept_metagraph_fp)
	incept_x = incept_graph.get_tensor_by_name('x:0')
	incept_preds = incept_graph.get_tensor_by_name('scores:0')
	incept_sess = tf.Session(graph=incept_graph)
	incept_saver.restore(incept_sess, args.incept_ckpt_fp)

	# Create summaries
	summary_graph = tf.Graph()
	with summary_graph.as_default():
		incept_mean = tf.placeholder(tf.float32, [])
		incept_std = tf.placeholder(tf.float32, [])
		summaries = [
				tf.summary.scalar('incept_mean', incept_mean),
				tf.summary.scalar('incept_std', incept_std)
		]
		summaries = tf.summary.merge(summaries)
	summary_writer = tf.summary.FileWriter(incept_dir)

	# Loop, waiting for checkpoints
	ckpt_fp = None
	_best_score = 0.
	while True:
		latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
		if latest_ckpt_fp != ckpt_fp:
			print('Incept: {}'.format(latest_ckpt_fp))

			sess = tf.Session(graph=gan_graph)

			gan_saver.restore(sess, latest_ckpt_fp)

			_step = sess.run(gan_step)

			_G_zs = []
			for i in range(0, args.incept_n, 100):
				_G_zs.append(sess.run(gan_G_z, {gan_z: _zs[i:i+100], gan_ngl: args.SpecGAN_ngl}))
			_G_zs = np.concatenate(_G_zs, axis=0)

			_preds = []
			for i in range(0, args.incept_n, 100):
				_preds.append(incept_sess.run(incept_preds, {incept_x: _G_zs[i:i+100]}))
			_preds = np.concatenate(_preds, axis=0)

			# Split into k groups
			_incept_scores = []
			split_size = args.incept_n // args.incept_k
			for i in range(args.incept_k):
				_split = _preds[i * split_size:(i + 1) * split_size]
				_kl = _split * (np.log(_split) - np.log(np.expand_dims(np.mean(_split, 0), 0)))
				_kl = np.mean(np.sum(_kl, 1))
				_incept_scores.append(np.exp(_kl))

			_incept_mean, _incept_std = np.mean(_incept_scores), np.std(_incept_scores)

			# Summarize
			with tf.Session(graph=summary_graph) as summary_sess:
				_summaries = summary_sess.run(summaries, {incept_mean: _incept_mean, incept_std: _incept_std})
			summary_writer.add_summary(_summaries, _step)

			# Save
			if _incept_mean > _best_score:
				score_saver.save(sess, os.path.join(incept_dir, 'best_score'), _step)
				_best_score = _incept_mean

			sess.close()

			print('Done')

			ckpt_fp = latest_ckpt_fp

		time.sleep(1)

	incept_sess.close()


"""
	Calculates and saves dataset moments
"""
def moments(fps, args):
	x = loader.get_batch(fps, 1, args._WINDOW_LEN, args.data_first_window, repeat=False)[0, :, 0]

	X = tf.contrib.signal.stft(x, 256, 128, pad_end=True)
	X_mag = tf.abs(X)
	X_lmag = tf.log(X_mag + args._LOG_EPS)

	_X_lmags = []
	with tf.Session() as sess:
		while True:
			try:
				_X_lmag = sess.run(X_lmag)
			except:
				break

			_X_lmags.append(_X_lmag)

	_X_lmags = np.concatenate(_X_lmags, axis=0)
	mean, std = np.mean(_X_lmags, axis=0), np.std(_X_lmags, axis=0)

	with open(os.path.join(args.train_dir, args.data_moments_file), 'wb') as f:
		pickle.dump((mean, std), f)

