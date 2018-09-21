import tensorflow as tf


def word_embedding(w, vocab_size=10, embedding_dim=256, train=False):
	W = tf.get_variable(name='W_embedding_matrix', 
						shape=[vocab_size, embedding_dim],
						dtype=tf.float32,
						initializer=tf.truncated_normal_initializer(stddev=1.0),
						trainable=True)
	b = tf.get_variable(name='b_embedding_matrix',
						shape=[embedding_dim], 
						dtype=tf.float32,
						initializer=tf.constant_initializer(0.0),
						trainable=True)
	w_embedding = tf.nn.xw_plus_b(w, W, b) # -> [batch_size, vocab_size] * [vocab_size, embedding_size] = [batch_size, embedding_size]
	return w_embedding


def conv2d_transpose(
		inputs,
		filters,
		kernel_len,
		stride=2,
		padding='same',
		upsample='zeros'):
	if upsample == 'zeros':
		return tf.layers.conv2d_transpose(
				inputs,
				filters,
				kernel_len,
				strides=(stride, stride),
				padding='same')
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
				padding='same')
	else:
		raise NotImplementedError()


"""
	Input: [None, 100]
	Output: [None, 128, 128, 1]
"""
def SpecGANGenerator(
		z,
		w=None,
		kernel_len=5,
		dim=64,
		use_batchnorm=False,
		upsample='zeros',
		train=False,
		cond=False):

	if cond == True and w == None: raise ValueError('Must feed a one-hot tensor as word condition for conditional training!')
	batch_size = tf.shape(z)[0]

	if use_batchnorm:
		batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
	else:
		batchnorm = lambda x: x

	# vector concatenation for conditional training
	# [64, 100] + [64, 256] -> [64, 356]
	if cond:
		output = tf.concat(values=[z, w], axis=1, name='z_w_concat')
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
def SpecGANDiscriminator(
		x,
		w=None,
		kernel_len=5,
		dim=64,
		use_batchnorm=False,
		cond=False):

	if cond == True and w == None: raise ValueError('Must feed a one-hot tensor as word condition for conditional training!')
	batch_size = tf.shape(x)[0]

	if use_batchnorm:
		batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
	else:
		batchnorm = lambda x: x

	# Layer 0
	# [128, 128, 1] -> [64, 64, 64]
	output = x
	with tf.variable_scope('downconv_0'):
		output = tf.layers.conv2d(output, dim, kernel_len, 2, padding='SAME')
	output = lrelu(output)

	# Layer 1
	# [64, 64, 64] -> [32, 32, 128]
	with tf.variable_scope('downconv_1'):
		output = tf.layers.conv2d(output, dim * 2, kernel_len, 2, padding='SAME')
		output = batchnorm(output)
	output = lrelu(output)

	# Layer 2
	# [32, 32, 128] -> [16, 16, 256]
	with tf.variable_scope('downconv_2'):
		output = tf.layers.conv2d(output, dim * 4, kernel_len, 2, padding='SAME')
		output = batchnorm(output)
	output = lrelu(output)

	# Layer 3
	# [16, 16, 256] -> [8, 8, 512]
	with tf.variable_scope('downconv_3'):
		output = tf.layers.conv2d(output, dim * 8, kernel_len, 2, padding='SAME')
		output = batchnorm(output)
	output = lrelu(output)

	# Layer 4
	# [8, 8, 512] -> [4, 4, 1024]
	with tf.variable_scope('downconv_4'):
		output = tf.layers.conv2d(output, dim * 16, kernel_len, 2, padding='SAME')
		output = batchnorm(output)
	output = lrelu(output)

	# vector concatenation for conditional training
	# [64, 256] -> [64, 1, 256] -> [64, 1, 1, 256] -> [64, 4, 4, 256] + [64, 4, 4, 1024]
	if cond:
		w = tf.expand_dims(w, axis=1)
		w = tf.expand_dims(w, axis=2)
		tiled_w = tf.tile(input=w, multiples=[1,4,4,1], name='tiled_w_embeddings') # -> This operation creates a new tensor by replicating input multiples times
		concat_w = tf.concat([output, tiled_w], axis=3, name='h3_concat') # -> shape: [64, 4, 4, 256] + [64, 4, 4, 1024] -> [64, 4, 4, 1024+256]
		output = tf.reshape(output, [batch_size, -1])
	else:
		output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16]) 

	# Connect to single logit
	with tf.variable_scope('output'):
		output = tf.layers.dense(output, 1)[:, 0]

	# Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training
	return output

"""
	# DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image, t_text_embedding, reuse=False):
		with tf.variable_scope(tf.get_variable_scope()) as scope:
			if reuse: scope.reuse_variables()

			h0 = ops.lrelu(ops.conv2d(input_=image, output_dim=self.df_dim, name='d_h0_conv')) # shape: (batch_size, 32, 32, 64)
			h1 = ops.lrelu(ops.conv2d(input_=h0, output_dim=self.df_dim*2, name='d_h1_conv')) # shape: (batch_size, 16, 16, 128)
			h2 = ops.lrelu(ops.conv2d(input_=h1, output_dim=self.df_dim*4, name='d_h2_conv')) # shape: (batch_size, 8, 8, 256)
			h3 = ops.lrelu(ops.conv2d(input_=h2, output_dim=self.df_dim*8, name='d_h3_conv')) # shape: (batch_size, 4, 4, 512)
			
			reduced_text_embeddings = ops.lrelu(ops.linear(input_=t_text_embedding, output_size=self.t_dim, name='d_embedding')) # shape: (batch_size, 256)
			reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, axis=1) # shape: (batch_size, 1, 256)
			reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, axis=2) # shape: (batch_size, 1, 1, 256)
			tiled_embeddings = tf.tile(input=reduced_text_embeddings, multiples=[1,4,4,1], name='tiled_embeddings') # shape: (batch_size, 4, 4, 256) -> This operation creates a new tensor by replicating input multiples times, tiling [a b c d] by [2] produces [a b c d a b c d]
			
			h3_concat = tf.concat([h3, tiled_embeddings], axis=3, name='h3_concat') # shape: (batch_size, 4, 4, 512+256)
			h3_new = ops.lrelu(ops.conv2d(input_=h3_concat, output_dim=self.df_dim*8, k_h=1, k_w=1, d_h=1, d_w=1, name='d_h3_conv_new')) # # shape: (batch_size, 4, 4, 512)
			
			output_layer = ops.linear(input_=tf.reshape(h3_new, [self.batch_size, -1]), output_size=1, name='d_h3_lin')
			return tf.nn.sigmoid(output_layer), output_layer
"""
