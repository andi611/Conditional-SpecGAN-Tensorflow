# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess_data.py ]
#   Synopsis     [ reprocess .wav files into clean and aligned .wav audios files ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import glob
import math
#-------------#
import numpy as np
import librosa
import librosa.display
#-------------#
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')


########
# PATH #
########
file_prefix = ''
in_dir = '../../data/sc09_wav/train'
out_dir = '../../data/sc09_preprocess_wav_energy/'
visualize_path = '../../data/visualize_preprocess_energy/'


############
# CONSTANT #
############
energy = True
sample_rate = 22050
window_size = 256


#################
# PROCESS AUDIO #
#################
"""
	Trim out the noisy parts in the audios,
	add begining and ending silence, and finally realign them.
"""
def process_audio(wavs, start_from=0, use_energy=False):

	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)
	if not os.path.isdir(visualize_path):
		os.makedirs(visualize_path)	
	if use_energy: print('Computing short time energy for endpoint detection!')

	for i, wav in enumerate(wavs):
		if i + 1 >= start_from:
			#---Load audio---#
			y, sr = librosa.load(wav)

			#---endpoint detection with energy---#
			threshold = 0.01
			if use_energy:
				print('Progress: %i/%i, computing energy...' % (i+1, len(wavs)), end='\r')
				En = _compute_short_time_energy(waveform=y, length_n=sr)
				#---end point detection with energy---#
				yt = _energy_based_truncation(waveform=y, energy=En, threshold=threshold)
				while librosa.get_duration(yt) > librosa.get_duration(y)*(2/3):
					threshold += 0.005
					yt = _energy_based_truncation(waveform=y, energy=En, threshold=threshold)
					if threshold >= 10: break
				print('Progress: %i/%i, computing energy...Done' % (i+1, len(wavs)), end='\r')
			if not use_energy or threshold >= 10:
				#---end point detection with db---#
				top_db = 10
				yt, index = librosa.effects.trim(y, top_db=top_db)
				while librosa.get_duration(yt) >= librosa.get_duration(y)*(2/3):
					top_db -= 1
					yt, index = librosa.effects.trim(y, top_db=top_db)
			print('Progress: %i/%i, Trimed audio from %.3f to %.3f' % (i+1, len(wavs), librosa.get_duration(y), librosa.get_duration(yt)))


			#---add beggining and ending silence---#
			beg_silence = np.zeros((int((sample_rate-len(yt))/2)))
			end_silence = np.zeros((sample_rate-len(beg_silence)-len(yt)))
			yt = np.concatenate([beg_silence, yt, end_silence], axis=0)
			assert len(yt) == sample_rate

			#---visualization---#
			plt.figure(figsize=(16, 4))
			if use_energy:
				plt.subplot(3, 1, 1)
				ax = plt.gca()
				ax.plot(np.arange(len(En)), En)
				ax.set_xlabel('Sample')
				ax.set_xlim([0, len(En)])
				plt.title('Short Time Energy Curve, En')
				plt.subplot(3, 1, 2)
				librosa.display.waveplot(y, sr=sr, color='r')
				plt.title('Original Waveform')
				plt.subplot(3, 1, 3)
			librosa.display.waveplot(yt, sr=sample_rate, color='tab:orange')
			plt.title('Processed Waveform')
			plt.tight_layout()
			wav = wav.split('/')[-1].split('.')[0]

			#---save---#
			plt.savefig(visualize_path + wav + '.jpeg')
			plt.close()
			wav = out_dir + wav + '.wav'
			librosa.output.write_wav(path=wav, y=yt, sr=sample_rate)


#########
# CHECK #
#########
"""
	Checks if all audios have been correctly processed,
	if not reprocess them.
"""
def check():
	redo_list = []
	
	#---get original file names---#
	wavs = glob.glob(os.path.join(in_dir, file_prefix) + '*.wav')
	for i in range(len(wavs)):
		wavs[i] = wavs[i].split('/')[-1] 
	
	#---get all preprocessed file names---#
	wavs_preprocess = glob.glob(os.path.join(out_dir, file_prefix) + '*.wav')
	for i in range(len(wavs_preprocess)):
		wavs_preprocess[i] = wavs_preprocess[i].split('/')[-1] 

	#---check for match and collect a redo list---#
	if len(wavs) != len(wavs_preprocess):
		for wav in wavs:
			if wav not in wavs_preprocess:
				redo_list.append(os.path.join(in_dir ,wav))
	
	#---reprocess---#
	if len(redo_list) != 0:
		process_audio(wavs=redo_list)
	print('Found %i audio files that needs to be processed, processing completed.' % len(redo_list))


##################
# HAMMING WINDOW #
##################
"""
	Hamming window for endpoint detection
"""
def _hamming_window(m, n, L):
	return 0.54 - 0.46 * math.cos( 2 * m * math.pi / L)


#####################
# SHORT TIME ENERGY #
#####################
"""
	Implemented based on the "Endpoint detection" section of the course DSP in NTUEE.
	Given an original waveform, compute the short time energy using the hamming window,
	and returns a list En, which is the computed energy curve.
"""
def _compute_short_time_energy(waveform, length_n, L=window_size):
	En = []
	for _ in range(int(L/2)): En.append(0)
	for n in range(int(L/2), len(waveform) - int(L/2)):
		E = 0.0
		for m in range(int(n-L/2), int(n+L/2)):
			E += waveform[m]**2 * _hamming_window(m, n, L)
		En.append(E)
	for _ in range(int(L/2)): En.append(0)
	return En


###########################
# ENERGY BASED TRUNCATION #
###########################
"""
	Traverse through the energy curve, and find a starting and ending sample
	according to the threshold,
	truncate the original waveform using the start and end sample index.
"""
def _energy_based_truncation(waveform, energy, threshold):
	start_idx = 0
	end_idx = len(waveform) - 1
	for idx, e in enumerate(energy):
		if e > threshold: start_idx = idx; break
	for idx, e in reversed(list(enumerate(energy))):
		if e > threshold: end_idx = idx; break
	waveform_t = waveform[start_idx:end_idx]
	return waveform_t


########
# MAIN #
########
def main():

	if not os.path.isdir(in_dir):
		raise ValueError('Please make sure there are .wav files in the directory: ', in_dir)

	start_from = 0 # -> preprocessing may terminate unexpectedly, manually change this index to resume the process

	wavs = glob.glob(os.path.join(in_dir, file_prefix) + '*.wav')
	process_audio(wavs=wavs, start_from=start_from, use_energy=energy)
	check()


if __name__ == '__main__':
	main()

