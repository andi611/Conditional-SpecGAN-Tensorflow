# conditional SpecGAN Tensorflow
a (conditional) audio synthesis generative adversarial network that generates spectrogram, which furthur synthesize raw waveform.

## Requirements: 
* **Tensorflow r1.10.1**
* Python 3.6
* numpy 1.14.5
* librosa 0.6.2
* tqdm 4.26.0
* matplotlib 2.2.3


## Introduction
This work is based on the original implementation of [SpecGAN](https://github.com/chrisdonahue/wavegan), where I furthur explore on conditioning SpecGAN training. Additionally, an energy based data preprocessing scheme is applied, which results in an improvement in audio quality.


## Build Dataset
1. Download training data [here](https://drive.google.com/open?id=102wZsFhhCOhq21UQT0cMH2oscLwyetrf)

2. Run './src/utils/preprocess_data.py' to process data or download the processed data [here](https://drive.google.com/file/d/1qyFRsSLI0cxyN10vFZnfcma4THPUulIN/view?usp=sharing)

3. Run './src/utils/visualize_wav.py' to visualize the processed clean data or download the results [here](https://drive.google.com/file/d/1vD_ufIBv5H7mCpmivPb5k9sBah2Ine9c/view?usp=sharing)

The preprocess result can be demonstrated by the following visualization:

4. Run './src/utils/make_tfrecord.py' to process .wav files into .tfrecord training ready files, or download the processed data [here](https://drive.google.com/file/d/1h1zJ3SiXafzE0Xn-7JWVeLtSckV3LrVT/view?usp=sharing)

5. Extract the .tgz file in step.4, and place them to the relevent path according to args.data_dir in ./src/config.py: 
```
data_dir='../data/sc09_preprocess_energy'
```
This default path can be modified by changing the '--data_dir option in './src/config.py'.


## Usage
1. Resume or train a new SpecGAN model by the following command:
```
python3 ./src/runner.py train
```

2. To inference and generate from a trained SpecGAN model, use the following command:
```
python3 ./src/runner.py generate
```

3. To train or generate from a conditional SpecGAN, use the following command (**Note: This feature is still under implementation and is not complete!**):
```
python3 ./src/runner.py train --conditional
python3 ./src/runner.py generate --conditional
```
