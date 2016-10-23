# Phoneme Recognition using RecNet


<!--- [![Build Status](https://travis-ci.org/joergfranke/rnnfwk.svg?branch=master)](https://travis-ci.org/joergfranke/rnnfwk) --->
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/joergfranke/phoneme_recognition/blob/master/LICENSE.txt)
[![Python](https://img.shields.io/badge/python-2.7-yellow.svg)](https://www.python.org/download/releases/2.7/)



This repository contains a automated phoneme recognition based on a recurrent neural network. The implementation uses
the [RecNet](https://github.com/joergfranke/recnet/) framework which is based on [Theano](http://deeplearning.net/software/theano/).
The used speech data set is the [TIMIT Acoustic-Phonetic Continuous Speech Corpus](https://catalog.ldc.upenn.edu/ldc93s1).

![alt tag](/images/example.png)


## Requirements

- Parts of the preprocess needs the SND file format library. For example it's installable via the APT interface.
```bash
sudo apt-get install libsndfile-dev
```
- This phoneme recognition model uses the rnnfwk RNN-Framework which needs to be installed. 
```bash
git clone https://github.com/joergfranke/rnnfwk.git
cd rnnfwk
python setup.py install
```
- There's a problem in 'scikits.audiolab's setup.py file.
Workaround: first install numpy, then requirements
- Furthermore the model requires the packages listed in the `requirements.txt`.
```bash
pip install -r requirements.txt
```



## How to use

### Step 1: Make data set

This step contains the whole pre-process and creates a data set in the form of two lists, one with sequences of
features (MFCC + log-energy + derivations) and one with corresponding sequences of targets (correct phonemes).
This pre-process is orientated on [Graves and Schmidhuber, 2005 ](ftp://ftp.idsia.ch/pub/juergen/nn_2005.pdf).
The data set gets stored in the klepto file format. Do the following for creating the data set:

1. Add path to the TIMIT corpus
2. Run `make_data_set.py`


### Step 2: Train recurrent neural network model

The second step contains the training of the model. Therefore it uses the [recnet](https://github.com/joergfranke/recnet/blob/master/README.md)
framework with a deep bidirectional LSTM architecture. Do the following for training the model:

1. Run `train_model.py`
2. Find the log file in the log folder

### Step 3: Evaluate exercised model

At least the traind model gets evaluated on the test set. The log loss and the rate of correct detected phonemes gets calculated.
A plot shows the input features, the true and the predicted phonemes.

1. Run `evaluate_model.py`





