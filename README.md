# Phoneme Recognition using Deep Bidirectional LSTMs


<!--- [![Build Status](https://travis-ci.org/joergfranke/rnnfwk.svg?branch=master)](https://travis-ci.org/joergfranke/rnnfwk) --->
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/joergfranke/phoneme_recognition/blob/master/LICENSE.txt)
[![Python](https://img.shields.io/badge/python-2.7-yellow.svg)](https://www.python.org/download/releases/2.7/)
[![Theano](https://img.shields.io/badge/theano-0.8.2-yellow.svg)](http://deeplearning.net/software/theano/)

## About
This repository contains a phoneme recognition based on a Deep Bidirectional Long Short Term Memory network.


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
- Furthermore the model requires the packages listed in the `requirements.txt`.
```bash
pip install -r requirements.txt
```



## How to use

### Step 1: Make data set

This step contains the whole preprocess and creates a data set in the form of two lists, one with sequences of features
and one with corresponding sequences of targets. The data set gets stored in the klepto file format. Do the following for creating the data set:

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





