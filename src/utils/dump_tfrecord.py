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
import sys
import glob
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write as wavwrite


###########
# _MAPPER #
###########
def _mapper(example_proto):
	features = {
			'samples': tf.FixedLenSequenceFeature([1], tf.float32, allow_missing=True),
			'label': tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)
	}
	example = tf.parse_single_example(example_proto, features)

	wav = example['samples'][:, 0]

	wav = wav[:16384]
	wav_len = tf.shape(wav)[0]
	wav = tf.pad(wav, [[0, 16384 - wav_len]])

	label = tf.reduce_join(example['label'], 0)
	return wav, label


########
# MAIN #
########
def main():
	try:
		out_dir, in_dir = sys.argv[1], sys.argv[2]
	except:
		in_dir = '../../data/sc09'
		out_dir = '../../data/sc09_wav'

	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)
		
	tfrecord_fps = glob.glob(os.path.join(in_dir, '*.tfrecord'))

	dataset = tf.data.TFRecordDataset(tfrecord_fps)
	dataset = dataset.map(_mapper)
	dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(1))
	x, y = dataset.make_one_shot_iterator().get_next()
	x, y = x[0], y[0]

	with tf.Session() as sess:
		i = 0
		while True:
			try:
				_x, _y = sess.run([x, y])
			except:
				break

			_x *= 32767.
			_x = np.clip(_x, -32767., 32767.)
			_x = _x.astype(np.int16)
			wavwrite(os.path.join(out_dir, '{}_{}.wav'.format(_y, str(i).zfill(5))), 16000, _x)
			i += 1


if __name__ == '__main__':
	main()

