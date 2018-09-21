# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ runner.py ]
#   Synopsis     [ main program that runs everything, execute this ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import sys
import glob
import pickle
from config import get_config
from generate import generate
from train import train, infer, preview, incept, moments


"""
	Main training program
"""
def main():	

	args = get_config()

	#---display model type---#
	print('-' * 80)
	print('# Training Conditional SpecGan!') if args.conditional else print('# Training SpecGan!')
	print('-' * 80)

	#---make train dir---#
	if args.conditional: args.train_dir = args.train_dir + '_cond'
	if not os.path.isdir(args.train_dir):
		os.makedirs(args.train_dir)

	#---save args---#
	with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
		f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

	#---make model kwarg dicts---#
	setattr(args, 'specgan_g_kwargs', {
			'kernel_len': args.specgan_kernel_len,
			'dim': args.specgan_dim,
			'use_batchnorm': args.specgan_batchnorm,
			'upsample': args.specgan_genr_upsample
	})
	setattr(args, 'specgan_d_kwargs', {
			'kernel_len': args.specgan_kernel_len,
			'dim': args.specgan_dim,
			'use_batchnorm': args.specgan_batchnorm
	})

	#---collect path to data---#
	if args.mode == 'train' or args.mode == 'moments':
		fps = glob.glob(os.path.join(args.data_dir, args.data_tfrecord_prefix) + '*.tfrecord')

	#---load moments---#
	if args.mode != 'moments' and args.data_moments_file is not None:
		while True:
			try:
				print('# Moments: Loading existing moments file...')
				with open(os.path.join(args.train_dir, args.data_moments_file), 'rb') as f:
					_mean, _std = pickle.load(f)
					break
			except:
				print('# Moments: Failed to load, computing new moments file...')
				moments(fps, args)
		setattr(args, 'data_moments_mean', _mean)
		setattr(args, 'data_moments_std', _std)

	#---run selected mode---#

	#---run generate mode--#
	if args.mode == 'train':
		infer(args, cond=args.conditional)
		train(fps, args, cond=args.conditional) 
	elif args.mode == 'generate':
		infer(args, cond=args.conditional)
		generate(args, cond=args.conditional)
	elif args.mode == 'moments':
		moments(fps, args)
	elif args.mode == 'preview':
		preview(args)
	elif args.mode == 'incept':
		incept(args)
	elif args.mode == 'infer':
		infer(args)
	else:
		raise NotImplementedError()


if __name__ == '__main__':
	main()


