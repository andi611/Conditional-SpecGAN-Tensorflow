# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ config.py ]
#   Synopsis     [ configuration settings ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""

import argparse


def get_config():
	parser = argparse.ArgumentParser()

	parser.add_argument('mode', type=str, choices=['train', 'moments', 'preview', 'incept', 'infer', 'generate'])
	parser.add_argument('--conditional', action='store_true', help='Train a conditional SpecGAN')

	data_args = parser.add_argument_group('Data')
	data_args.add_argument('--data_dir', type=str, help='Data directory')
	data_args.add_argument('--data_tfrecord_prefix', type=str, help='Prefix of the .tfrecord files')
	data_args.add_argument('--data_first_window', action='store_true', help='If set, only use the first window from each audio example')
	data_args.add_argument('--data_moments_fp', type=str, help='Path to store and retrieve the data moments .pkl file')

	specgan_args = parser.add_argument_group('SpecGAN')
	specgan_args.add_argument('--specgan_kernel_len', type=int, help='Length of square 2D filter kernels')
	specgan_args.add_argument('--specgan_dim', type=int, help='Dimensionality multiplier for model of G and D')
	specgan_args.add_argument('--specgan_batchnorm', action='store_true', help='Enable batchnorm')
	specgan_args.add_argument('--specgan_disc_nupdates', type=int, help='Number of discriminator updates per generator update')
	specgan_args.add_argument('--specgan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'], help='Which GAN loss to use')
	specgan_args.add_argument('--specgan_genr_upsample', type=str, choices=['zeros', 'nn', 'lin', 'cub'], help='Generator upsample strategy')
	specgan_args.add_argument('--specgan_ngl', type=int, help='Number of Griffin-Lim iterations')
	specgan_args.add_argument('--specgan_word_embedding_dim', type=int, help='Dimension for word conditional vectors')

	train_args = parser.add_argument_group('Train')
	train_args.add_argument('--train_dir', type=str, help='Training directory')
	train_args.add_argument('--train_batch_size', type=int, help='Batch size')
	train_args.add_argument('--train_max_step', type=int, help='Maximum training steps before terminating training')
	train_args.add_argument('--train_save_secs', type=int, help='How often to save model')
	train_args.add_argument('--train_summary_secs', type=int, help='How often to report summaries')
	train_args.add_argument('--train_display_step', type=int, help='How often to display training log')
	train_args.add_argument('--train_save_log_step', type=int, help='How often to save training log')

	preview_args = parser.add_argument_group('Preview')
	preview_args.add_argument('--preview_n', type=int, help='Number of samples to preview')

	incept_args = parser.add_argument_group('Incept')
	incept_args.add_argument('--incept_metagraph_fp', type=str, help='Inference model for inception score')
	incept_args.add_argument('--incept_ckpt_fp', type=str, help='Checkpoint for inference model')
	incept_args.add_argument('--incept_n', type=int, help='Number of generated examples to test')
	incept_args.add_argument('--incept_k', type=int, help='Number of groups to test')

	preview_args = parser.add_argument_group('Generate')
	preview_args.add_argument('--generate_dir', type=str, help='Generation directory')
	preview_args.add_argument('--generate_num', type=int, help='Number of samples to generate')
	preview_args.add_argument('--generate_visualize_num', type=int, help='Number of samples to generate and visualize')

	constant_args = parser.add_argument_group('Constants')
	constant_args.add_argument('--_VOCAB_SIZE', type=int, help='Expected vocabulary size')
	constant_args.add_argument('--_FS', type=int, help='Frequency')
	constant_args.add_argument('--_WINDOW_LEN', type=int, help='Window length')
	constant_args.add_argument('--_D_Z ', type=int, help='Dimension of noise z')
	constant_args.add_argument('--_LOG_EPS', type=float, help='Log eps constant')
	constant_args.add_argument('--_CLIP_NSTD', type=float, help='Clip stantard normalization')

	parser.set_defaults(
		#---data---#
		data_dir='../data/sc09_preprocess',
		data_tfrecord_prefix='sc09',
		data_first_window=False,
		data_moments_file='moments.pkl',
		#---specgan---#
		specgan_kernel_len=5,
		specgan_dim=64,
		specgan_batchnorm=False,
		specgan_disc_nupdates=5,
		specgan_loss='wgan-gp',
		specgan_genr_upsample='zeros',
		specgan_ngl=16,
		specgan_word_embedding_dim=256,
		#---train---#
		train_dir='../train',
		train_batch_size=64,
		train_max_step=300000,
		train_save_secs=300,
		train_summary_secs=300,
		train_display_step=20,
		train_save_log_step=100,
		#---preview---#
		preview_n=32,
		#---incept---#
		incept_metagraph_fp='../eval/inception/infer.meta',
		incept_ckpt_fp='../eval/inception/best_acc-103005',
		incept_n=5000,
		incept_k=10,
		#---generate---#
		generate_dir='../generate',
		generate_num=50,
		generate_visualize_num=5,
		#---constant---#
		_VOCAB_SIZE=10,
		_FS=16000,
		_WINDOW_LEN=16384,
		_D_Z=100,
		_LOG_EPS=1e-6,
		_CLIP_NSTD=3.0)

	args = parser.parse_args()
	return args

