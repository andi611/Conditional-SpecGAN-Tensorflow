# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ visualize_wav.py ]
#   Synopsis     [ generate .jpeg images from .wav audios ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import glob
#-------------#
import numpy as np
#-------------#
import librosa
import librosa.display
#-------------#
import matplotlib
import matplotlib.pyplot as plt
#-------------#
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})


########
# PATH #
########
in_dir = '../../data/sc09_wav/train'
out_dir = '../../data/visualize/'


########
# MAIN #
########
def main():

	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)

	wavs = glob.glob(os.path.join(in_dir, 'b') + '*.wav')

	for i, wav in enumerate(wavs):
		x, fs = librosa.load(wav)
		fig = plt.figure(figsize=(16, 4))
		librosa.display.waveplot(x, sr=fs)
		fig.tight_layout()
		wav = wav.split('/')[-1].split('.')[0]
		fig.savefig(out_dir + wav + '.jpeg')
		plt.close(fig)
		plt.cla()
		print('Progress: %i/%i.' % (i+1, len(wavs)), end='\r')

	print('Progress: %i/%i. Complete.' % (i+1, len(wavs)))


if __name__ == '__main__':
	main()

