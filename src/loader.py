# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dump_tfrecord.py ]
#   Synopsis     [ generate .wav audios from .tfrecord files ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import pickle
import tensorflow as tf


#############
# GET_BATCH #
#############
"""
	fps: List of tfrecords
	batch_size: Resultant batch size
	window_len: Size of slice to take from each example
	first_window: If true, always take the first window in the example, otherwise take a random window
	repeat: If false, only iterate through dataset once
	labels: If true, return (x, y), else return x
	buffer_size: Number of examples to queue up (larger = more random)
"""
def get_batch(
		fps,
		batch_size,
		window_len,
		first_window=False,
		repeat=True,
		labels=False,
		buffer_size=16384):

	def _mapper(example_proto):
		features = {'samples': tf.FixedLenSequenceFeature([1], tf.float32, allow_missing=True)}
		if labels:
			features['label'] = tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)

		example = tf.parse_single_example(serialized=example_proto, features=features)
		wav = example['samples']
		if labels:
			label = tf.reduce_join(example['label'], 0)

		if first_window:
			# Use first window
			wav = wav[:window_len]
		else:
			# Select random window
			wav_len = tf.shape(wav)[0]

			start_max = wav_len - window_len
			start_max = tf.maximum(start_max, 0)

			start = tf.random_uniform([], maxval=start_max + 1, dtype=tf.int32)

			wav = wav[start:start+window_len]

		wav = tf.pad(wav, [[0, window_len - tf.shape(wav)[0]], [0, 0]])

		wav.set_shape([window_len, 1])

		if labels:
			return wav, label
		else:
			return wav

	dataset = tf.data.TFRecordDataset(fps)
	dataset = dataset.map(_mapper)
	if repeat:
		dataset = dataset.shuffle(buffer_size=buffer_size)
	dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
	if repeat:
		dataset = dataset.repeat()
	iterator = dataset.make_one_shot_iterator()

	return iterator.get_next()


###################
# LABEL TO TENSOR #
###################
"""
	Convert string labels to tensorflow one-hot tensors by:
	first computing word-to-index mapper from data,
	then convert indexs to one-hot vectors.
	Use in training.
"""
def label_to_tensor(label, args, fps):
	mapper = _parse_labels(fps, mapper_path=args.train_dir)
	tf_mapper = _get_tf_mapper(mapper) # -->> this mapper needs to be initialized by 'sess.run(tf.tables_initializer())' if ran without MonitoredSession
	
	if len(mapper['word2idx']) != args._VOCAB_SIZE: raise ValueError('Number of vocabulary inconsistent!')
	label_idx = tf_mapper.lookup(keys=label)
	label_one_hot = tf.one_hot(indices=label_idx, depth=args._VOCAB_SIZE)

	#---generate wrong label---#
	random_idx = tf.random_uniform(dtype=tf.int32, minval=0, maxval=10, shape=[args.train_batch_size]) # First sample from [minval, maxval)
	wrong_label = tf.where(random_idx >= label_idx, random_idx+1, random_idx) # Increment for values == the original label
	wrong_one_hot = tf.one_hot(indices=wrong_label, depth=args._VOCAB_SIZE)
	
	return label_one_hot, wrong_one_hot


######################
# IDX TO CATEGORICAL #
######################
"""
	Convert idx list to tensorflow one-hot tensors
	Use in generation.
"""
def idx_to_categorical(idx, args):
	tensor = tf.convert_to_tensor(value=idx)
	one_hot = tf.one_hot(indices=idx, depth=args._VOCAB_SIZE)
	return one_hot


###############
# PARSE LABEL #
###############
"""
	Read labels from .tfrecord files and parse them into 'word to idx' dictionaries
"""
def _parse_labels(fps, mapper_path):

	mapper_path = os.path.join(mapper_path, 'mapper.pkl')

	#---load if possible---#
	if os.path.isfile(mapper_path):
		mapper = pickle.load(open(mapper_path, 'rb'))
		print('# Mapper: Loading existing mapper file...')
		return mapper

	def _read_and_decode(reader, tf_record_filename_queue):
		_, tf_record_serialized = reader.read(tf_record_filename_queue)
		example = tf.parse_single_example(
								serialized=tf_record_serialized, 
								features={'label': tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)})
		return tf.reduce_join(example['label'], 0)

	print('# Mapper: Failed to load, computing new mapper file...')
	with tf.Session() as sess:
		#---setting up tf op---#
		reader = tf.TFRecordReader()
		tf_record_filename_queue = tf.train.string_input_producer(fps, num_epochs=1)
		label_op = _read_and_decode(reader, tf_record_filename_queue)
		count_op = reader.num_records_produced()
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		#---initializing session---#
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		try:
			#---parse labels---#
			labels = []
			while True:
				label = str(sess.run(label_op).decode("utf-8")).strip()
				print('Iteration %i, unique labels found: ' % int(sess.run(count_op)), labels, end='\r')
				if label not in labels:
					labels.append(label)

		except tf.errors.OutOfRangeError as e:
			print()
			coord.request_stop(e)

		finally:
			#---build mapping dictionaires---#
			word2idx = {}
			for i, label in enumerate(labels):
				word2idx[label] = i
			idx2word = { i : w for w, i in word2idx.items() }
			mapper = {'word2idx': word2idx, 'idx2word': idx2word}
			pickle.dump(mapper, open(mapper_path, 'wb'), True)

			#---terminating---#
			print('Parsing complete! Number of unique tags found: ', len(labels))
			coord.request_stop()
			coord.join(threads)
			tf_record_filename_queue.close(cancel_pending_enqueues=True)
			return mapper


#########################
# GET TENSORFLOW MAPPER #
#########################
"""
	Create tf lookup hash table from python dictionaires
"""
def _get_tf_mapper(mapper):
	keys = []
	values = []
	for w, i in mapper['word2idx'].items():
		keys.append(w)
		values.append(i)
	tf_mapper = tf.contrib.lookup.HashTable(
							initializer=tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
							default_value=-1)
	return tf_mapper


