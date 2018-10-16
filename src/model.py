# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ GAN model architecture ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import tensorflow as tf


"""
	tf variable initializer
"""
def variable_initializer(initializer):
	if initializer == 'orthogonal':
		return tf.orthogonal_initializer(gain=1.0, seed=None)
	elif initializer == 'default':
		return None
	else:
		return NotImplementedError()


"""
	Input: n-class one-hot vector
	Output: word embedding vector
"""
def word_embedding(word, vocab_size=10, embedding_dim=256, train=False):
	W = tf.get_variable(name='W_embedding_matrix', 
						shape=[vocab_size, embedding_dim],
						dtype=tf.float32,
						initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=7942089),
						trainable=train)
	b = tf.get_variable(name='b_embedding_matrix',
						shape=[embedding_dim], 
						dtype=tf.float32,
						initializer=tf.constant_initializer(0.0),
						trainable=train)
	w_embedding = tf.nn.xw_plus_b(word, W, b) # -> [batch_size, vocab_size] * [vocab_size, embedding_size] = [batch_size, embedding_size]
	return w_embedding


"""
	transpose convolution
"""
def conv2d_transpose(
		inputs,
		filters,
		kernel_len,
		stride=2,
		padding='same',
		upsample='zeros',
		initializer='default'):
	if upsample == 'zeros':
		return tf.layers.conv2d_transpose(
				inputs,
				filters,
				kernel_len,
				strides=(stride, stride),
				padding='same',
				kernel_initializer=variable_initializer(initializer))
	elif upsample in ['nn', 'linear', 'cubic']:
		batch_size = tf.shape(inputs)[0]
		_, h, w, nch = inputs.get_shape().as_list()

		x = inputs

		if upsample == 'nn':
			upsampler = tf.image.resize_nearest_neighbor
		elif upsample == 'linear':
			upsampler = tf.image.resize_bilinear
		else:
			upsampler = tf.image.resize_bicubic

		x = upsampler(x, [h * stride, w * stride])
		
		return tf.layers.conv2d(
				x,
				filters,
				kernel_len,
				strides=(1, 1),
				padding='same',
				kernel_initializer=variable_initializer(initializer))
	else:
		raise NotImplementedError()


"""
	Input: [None, 100]
	Output: [None, 128, 128, 1]
"""
def Spec_GAN_Generator(
		z,
		word=None,
		kernel_len=5,
		dim=64,
		use_batchnorm=False,
		upsample='zeros',
		initializer='default',
		train=False,
		cond=False):

	if cond == True and word == None: raise ValueError('Must feed a one-hot tensor as word condition for conditional training!')
	batch_size = tf.shape(z)[0]

	if use_batchnorm:
		batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
	else:
		batchnorm = lambda x: x

	# vector concatenation for conditional training
	# [64, 100] + [64, 256] -> [64, 356]
	if cond:
		output = tf.concat(values=[z, word], axis=1, name='z_w_concat')
	else:
		output = z
	
	# FC and reshape for convolution
	# [100] or [356] -> [4, 4, 1024]
	with tf.variable_scope('z_project'):
		output = tf.layers.dense(output, 4 * 4 * dim * 16)
		output = tf.reshape(output, [batch_size, 4, 4, dim * 16])
		output = batchnorm(output)
	output = tf.nn.relu(output)

	# Layer 0
	# [4, 4, 1024] -> [8, 8, 512]
	with tf.variable_scope('upconv_0'):
		output = conv2d_transpose(output, dim * 8, kernel_len, 2, upsample=upsample)
		output = batchnorm(output)
	output = tf.nn.relu(output)

	# Layer 1
	# [8, 8, 512] -> [16, 16, 256]
	with tf.variable_scope('upconv_1'):
		output = conv2d_transpose(output, dim * 4, kernel_len, 2, upsample=upsample)
		output = batchnorm(output)
	output = tf.nn.relu(output)

	# Layer 2
	# [16, 16, 256] -> [32, 32, 128]
	with tf.variable_scope('upconv_2'):
		output = conv2d_transpose(output, dim * 2, kernel_len, 2, upsample=upsample)
		output = batchnorm(output)
	output = tf.nn.relu(output)

	# Layer 3
	# [32, 32, 128] -> [64, 64, 64]
	with tf.variable_scope('upconv_3'):
		output = conv2d_transpose(output, dim, kernel_len, 2, upsample=upsample)
		output = batchnorm(output)
	output = tf.nn.relu(output)

	# Layer 4
	# [64, 64, 64] -> [128, 128, 1]
	with tf.variable_scope('upconv_4'):
		output = conv2d_transpose(output, 1, kernel_len, 2, upsample=upsample)
	output = tf.nn.tanh(output)

	# Automatically update batchnorm moving averages every time G is used during training
	if train and use_batchnorm:
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		if len(update_ops) != 10:
			raise Exception('Other update ops found in graph')
		with tf.control_dependencies(update_ops):
			output = tf.identity(output)

	return output


def lrelu(inputs, alpha=0.2):
	return tf.maximum(alpha * inputs, inputs)


"""
	Input: [None, 128, 128, 1]
	Output: [None] (linear) output
"""
def Spec_GAN_Discriminator(
		x,
		word=None,
		kernel_len=5,
		dim=64,
		use_batchnorm=False,
		initializer='default',
		cond=False):

	if cond == True and word == None: raise ValueError('Must feed a one-hot tensor as word condition for conditional training!')
	batch_size = tf.shape(x)[0]

	if use_batchnorm:
		batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
	else:
		batchnorm = lambda x: x

	# Layer 0
	# [128, 128, 1] -> [64, 64, 64]
	output = x
	with tf.variable_scope('downconv_0'):
		output = tf.layers.conv2d(output, dim, kernel_len, 2,
								padding='SAME',
								kernel_initializer=variable_initializer(initializer))
	output = tf.nn.leaky_relu(output)

	# Layer 1
	# [64, 64, 64] -> [32, 32, 128]
	with tf.variable_scope('downconv_1'):
		output = tf.layers.conv2d(output, dim * 2, kernel_len, 2, 
								padding='SAME',
								kernel_initializer=variable_initializer(initializer))
		output = batchnorm(output)
	output = tf.nn.leaky_relu(output)

	# Layer 2
	# [32, 32, 128] -> [16, 16, 256]
	with tf.variable_scope('downconv_2'):
		output = tf.layers.conv2d(output, dim * 4, kernel_len, 2, 
								padding='SAME',
								kernel_initializer=variable_initializer(initializer))
		output = batchnorm(output)
	output = tf.nn.leaky_relu(output)

	# Layer 3
	# [16, 16, 256] -> [8, 8, 512]
	with tf.variable_scope('downconv_3'):
		output = tf.layers.conv2d(output, dim * 8, kernel_len, 2, 
								padding='SAME',
								kernel_initializer=variable_initializer(initializer))
		output = batchnorm(output)
	output = tf.nn.leaky_relu(output)

	# Layer 4
	# [8, 8, 512] -> [4, 4, 1024]
	with tf.variable_scope('downconv_4'):
		output = tf.layers.conv2d(output, dim * 16, kernel_len, 2, 
								padding='SAME',
								kernel_initializer=variable_initializer(initializer))
		output = batchnorm(output)
	output = tf.nn.leaky_relu(output)

	# vector concatenation for conditional training
	# [64, 256] -> [64, 1, 256] -> [64, 1, 1, 256] -> [64, 4, 4, 256] + [64, 4, 4, 1024]
	if cond:
		word = tf.expand_dims(word, axis=1)
		word = tf.expand_dims(word, axis=2)
		tiled_w = tf.tile(input=word, multiples=[1,4,4,1], name='tiled_w_embeddings') # -> This operation creates a new tensor by replicating input multiples times
		concat_w = tf.concat([output, tiled_w], axis=3, name='h3_concat') # -> shape: [64, 4, 4, 256] + [64, 4, 4, 1024] -> [64, 4, 4, 1024+256]
		output = tf.reshape(output, [batch_size, -1])
	else:
		output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16]) 

	# Connect to single logit
	with tf.variable_scope('output'):
		output = tf.layers.dense(output, 1, kernel_initializer=variable_initializer(initializer))[:, 0]

	# Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training
	return output


