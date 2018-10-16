# conditional SpecGAN
**A (conditional) audio synthesis generative adversarial network that generates spectrogram, which furthur synthesize raw waveform, implementation in Tensorflow**
![](https://github.com/andi611/conditional_SpecGAN_Tensorflow/blob/master/data/model.png)

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
* Download training data [here](https://drive.google.com/open?id=102wZsFhhCOhq21UQT0cMH2oscLwyetrf)

* Run './src/utils/preprocess_data.py' to process data or download the processed data [here](https://drive.google.com/file/d/1qyFRsSLI0cxyN10vFZnfcma4THPUulIN/view?usp=sharing)

* Run './src/utils/visualize_wav.py' to visualize the processed clean data or download the results [here](https://drive.google.com/file/d/1vD_ufIBv5H7mCpmivPb5k9sBah2Ine9c/view?usp=sharing)

* The preprocess result can be demonstrated by the following visualization:
![](https://github.com/andi611/conditional_SpecGAN_Tensorflow/blob/master/data/preprocess_demo.jpeg)

* Run './src/utils/make_tfrecord.py' to process .wav files into .tfrecord training ready files, or download the processed data [here](https://drive.google.com/file/d/1h1zJ3SiXafzE0Xn-7JWVeLtSckV3LrVT/view?usp=sharing)

* Extract the .tgz file in step.4, and place them to the relevent path according to args.data_dir in ./src/config.py: 
```
data_dir='../data/sc09_preprocess_energy'
```
This default path can be modified by changing the '--data_dir option in './src/config.py'.


## Usage
* Resume or train a new SpecGAN model by the following command:
```
python3 ./src/runner.py train
```

* To inference and generate from a trained SpecGAN model, use the following command:
```
python3 ./src/runner.py generate
```

* To train or generate from a conditional SpecGAN, use the following command (**Note: This feature is still under implementation and is not complete!**):
```
python3 ./src/runner.py train --conditional
python3 ./src/runner.py generate --conditional
```
