# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ generate.py ]
#   Synopsis     [ Generate wavefroms from trained model as .jpeg images and .wav audios ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import glob
import numpy as np
import tensorflow as tf
#-------------#
import librosa
import librosa.display
import matplotlib.pyplot as plt
#-------------#
import loader
from config import get_config
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})


def generate(args, cond=False):	


	#---get model name---#
	mdls = glob.glob(os.path.join(args.train_dir, '*.meta'))
	mdl_name = sorted(mdls)[-1].split('-')[-1].split('.')[0]
	mdl_name = 'cond_SpecGAN_' + mdl_name if args.conditional else 'SpecGAN_' + mdl_name
	if args.conditional: args.generate_dir = args.generate_dir + '_cond' 
	mdl_dir = os.path.join(args.generate_dir, mdl_name)

	#---make generate dir---#
	if not os.path.isdir(mdl_dir):
		os.makedirs(mdl_dir)
	
	#---load the graph---#
	tf.reset_default_graph()
	saver = tf.train.import_meta_graph(os.path.join(args.train_dir, 'infer/infer.meta'))
	graph = tf.get_default_graph()
	sess = tf.InteractiveSession()
	saver.restore(sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=args.train_dir))

	#---create 50 random latent vectors z---#
	if args.SpecGAN_prior_noise == 'uniform':
		_z = np.random.uniform(low=-1., high=1.0, size=(args.generate_num, args._D_Z))
	elif args.SpecGAN_prior_noise == 'normal':
		_z = np.random.normal(loc=0., scale=1.0, size=(args.generate_num, args._D_Z))
	else:
		raise NotImplementedError()

	#---get tensors---#
	z = graph.get_tensor_by_name('z:0')
	ngl = graph.get_tensor_by_name('ngl:0')
	G_z = graph.get_tensor_by_name('G_z:0')
	feed_dict = {z: _z, ngl: args.SpecGAN_ngl}
	if cond:
		label = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'] * 5
		mapper = loader._parse_labels(fps=None, mapper_path=args.train_dir)
		label = [mapper['word2idx'][l] for l in label]
		cond_word_label = graph.get_tensor_by_name('cond_word_label:0')
		feed_dict[cond_word_label] = label
		
	#---synthesize G(z)---#
	_G_z = sess.run(G_z, feed_dict)

	# Show shape
	print('z shape: ',  np.shape(_z))
	print('G_z shape: ', np.shape(_G_z))
	if cond: print('w_input_shape: ', np.shape(label))

	# Visualize generated wavefrom
	visualize_list = _G_z[:args.generate_visualize_num]
	for i, v in enumerate(visualize_list):
		fig = plt.figure(figsize=(16, 4))
		librosa.display.waveplot(np.squeeze(v), sr=16000)
		fig.tight_layout()
		fig.savefig(os.path.join(mdl_dir, 'generate_') + str(i) + '.jpeg')
		plt.close(fig)
		plt.cla()

	# Save results
	gen = np.concatenate([_g_z for _g_z in _G_z], axis=0)
	librosa.output.write_wav(path=os.path.join(mdl_dir, 'generate.wav'), y=gen, sr=16000)
	print('Generation of %i samples and visualization of %i samples complete.' % (args.generate_num, args.generate_visualize_num))
	
